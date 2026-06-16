#%% Part 0 import package and Global Parameters
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import threshold
from torch.optim import Adam
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import trange

import numpy as np
import random
import copy
import math
from loguru import logger
import itertools
import wandb

from agents.fisor_2024.models.diffusion import Diffusion, DiffusionUnCond, DiffusionV1, DiffusionV2, FlowMatching, \
    FlowMatchingUnCond
from agents.fisor_2024.models.networks import mlp, EnsembleValue, EnsembleQCritic

from torch.distributions import Normal, Categorical, TransformedDistribution, TanhTransform


# LOG_SIG_MAX = 2
# LOG_SIG_MIN = -20
# epsilon = 1e-6

#%% Part 1 Global Function Definition
def setup_seed(seed=1024): # After doing this, the Training results will always be the same for the same seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info(f"Seed {seed} has been set for all modules!")

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def soft_update(target, source, tau): # Target will be updated but Source will not change
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):      # Target will be updated but Source will not change
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def cost2h(cost, cost_scale):
    h = torch.where(cost > 0, torch.tensor(cost_scale).to(cost.device),
                    torch.tensor(-1.0).to(cost.device))
    return h

def expectile_loss(u, tau=0.8):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

def exp_schedule_expectile_temp(step, max_step, start=0.5, end=0.99):
    # Exponential growth formula
    scale = (end - start)
    return start + scale * (1 - math.exp(-20 * step / max_step))

def linear_schedule_expectile_temp(step, max_step, start=0.5, end=0.99):
    # Linear growth formula
    scale = (end - start) / max_step
    return start + scale * step

#%% Part 2 Network Definition
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLP_POLICY(nn.Module):
    def __init__(
            self,
            hidden_dims: list,
            activations: callable = nn.GELU,
            activate_final: bool = False
    ):
        super().__init__()
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if i < len(hidden_dims) - 2 or activate_final:
                layers.append(activations())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Policy(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_dims: list,
            log_std_min: float = -20,
            log_std_max: float = 2,
            tanh_squash_distribution: bool = False,
            state_dependent_std: bool = True,
            final_fc_init_scale: float = 1e-2
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.tanh_squash = tanh_squash_distribution
        self.state_dependent_std = state_dependent_std

        # CREATE BASE NET
        hidden_dims = [obs_dim,] + hidden_dims
        self.base_net = MLP_POLICY(
            hidden_dims=hidden_dims,
            activate_final=True
        )

        # MEAN STD OUTPUT LAYER
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        nn.init.uniform_(self.mean_layer.weight, -final_fc_init_scale, final_fc_init_scale)
        nn.init.zeros_(self.mean_layer.bias)

        # 标准差处理
        if state_dependent_std:
            self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
            nn.init.uniform_(self.log_std_layer.weight, -final_fc_init_scale, final_fc_init_scale)
            nn.init.zeros_(self.log_std_layer.bias)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
            self,
            observations: torch.Tensor,
            temperature: float = 1.0
    ) -> torch.distributions.Distribution:
        features = self.base_net(observations)

        # 计算均值
        mean = self.mean_layer(features)

        # 计算对数标准差
        if self.state_dependent_std:
            log_std = self.log_std_layer(features)
        else:
            log_std = self.log_std.expand_as(mean)

        # 限制标准差范围
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std) * temperature + 1e-6

        # 创建高斯分布
        base_dist = Normal(mean, std)

        # 如果需要 tanh 变换
        if self.tanh_squash:
            transforms = TanhTransform()
            dist = TransformedDistribution(base_dist, transforms)
            # 添加 mode 方法
            dist.mode = lambda: torch.tanh(base_dist.mean)
            return dist

        return base_dist

class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, device, t_dim=16):
        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)


class MLPResNetBlock(nn.Module):
    """MLPResNet block."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, act: callable,
                 dropout_rate: float = None, use_layer_norm: bool = False):
        super(MLPResNetBlock, self).__init__()
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        # Define layers
        self.dense1 = nn.Linear(input_dim,  hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)

        if input_dim != output_dim:
            self.res = nn.Linear(input_dim, output_dim)
        else:
            self.res = nn.Identity()

    def forward(self, x):
        residual = self.res(x)

        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            x = self.dropout(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)

        return residual + x


class MLPResNet(nn.Module):
    def __init__(self, state_dim, action_dim, device, num_blocks, hidden_dim, t_dim=64):
        super(MLPResNet, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.Mish(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # CONDITION CONTAINS REWARD AND RETURN
        self.r_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim*4),
            nn.Mish(),
            nn.Linear(hidden_dim*4, t_dim),
        )
        self.c_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim*4),
            nn.Mish(),
            nn.Linear(hidden_dim*4, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim*3

        self.first_block = MLPResNetBlock(input_dim, hidden_dim*4, hidden_dim,
                                          nn.Mish(), dropout_rate=0.0, use_layer_norm=False)

        self.resnet = nn.ModuleList(
            [MLPResNetBlock(hidden_dim, hidden_dim*4, hidden_dim, nn.Mish(), dropout_rate=0.1, use_layer_norm=True)
             for _ in range(num_blocks)
             ])

        self.output_layer = nn.Sequential(
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out output layers:
        nn.init.constant_(self.output_layer[-1].weight, 0)
        nn.init.constant_(self.output_layer[-1].bias, 0)

    def forward(self, x, time, state, cond, resnet_skip=True):
        r_embedding = self.r_embedding(cond[:,0:1])
        c_embedding = self.c_embedding(cond[:,1:2])

        r_uncond_mask = cond[:, 2:3]
        c_uncond_mask = cond[:, 3:4]
        r_embedding = r_uncond_mask * r_embedding
        c_embedding = c_uncond_mask * c_embedding

        t = self.time_mlp(time)
        t = torch.cat([t, r_embedding, c_embedding], dim=-1)
        _x = torch.cat([state, x, t], dim=-1)
        x = self.first_block(_x)
        residual = x

        for block in self.resnet:
            x = block(x)

        if resnet_skip:
            x = self.output_layer(x + residual)
        else:
            x = self.output_layer(x)

        return x


class MLPResNetV1(nn.Module):
    def __init__(self, state_dim, action_dim, device, num_blocks, hidden_dim, t_dim=64):
        super(MLPResNetV1, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.GELU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # CONDITION CONTAINS REWARD AND RETURN
        self.r_embedding = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, t_dim),
        )
        self.c_embedding = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim*3

        self.first_block = MLPResNetBlock(input_dim, hidden_dim*4, hidden_dim,
                                          nn.Mish(), dropout_rate=0.0, use_layer_norm=False)

        self.resnet = nn.ModuleList(
            [MLPResNetBlock(hidden_dim, hidden_dim*4, hidden_dim, nn.Mish(), dropout_rate=0.1, use_layer_norm=True)
             for _ in range(num_blocks)
             ])

        self.output_layer = nn.Sequential(
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out output layers:
        nn.init.constant_(self.output_layer[-1].weight, 0)
        nn.init.constant_(self.output_layer[-1].bias, 0)

    def forward(self, x, time, state, cond, resnet_skip=True):
        r_embedding = self.r_embedding(cond[:,0:1])
        c_embedding = self.c_embedding(cond[:,1:2])

        t = self.time_mlp(time)
        t = torch.cat([t, r_embedding, c_embedding], dim=-1)
        _x = torch.cat([state, x, t], dim=-1)
        x = self.first_block(_x)
        residual = x

        for block in self.resnet:
            x = block(x)

        if resnet_skip:
            x = self.output_layer(x + residual)
        else:
            x = self.output_layer(x)

        return x


class MLPResNetV2(nn.Module):
    def __init__(self, state_dim, action_dim, device, num_blocks, hidden_dim, t_dim=64):
        super(MLPResNetV2, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.GELU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # CONDITION CONTAINS REWARD AND RETURN
        self.r_embedding = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim*2

        self.first_block = MLPResNetBlock(input_dim, hidden_dim*4, hidden_dim,
                                          nn.Mish(), dropout_rate=0.0, use_layer_norm=False)

        self.resnet = nn.ModuleList(
            [MLPResNetBlock(hidden_dim, hidden_dim*4, hidden_dim, nn.Mish(), dropout_rate=0.1, use_layer_norm=True)
             for _ in range(num_blocks)
             ])

        self.output_layer = nn.Sequential(
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out output layers:
        nn.init.constant_(self.output_layer[-1].weight, 0)
        nn.init.constant_(self.output_layer[-1].bias, 0)

    def forward(self, x, time, state, cond, resnet_skip=True):
        r_embedding = self.r_embedding(cond)

        t = self.time_mlp(time)
        t = torch.cat([t, r_embedding], dim=-1)
        _x = torch.cat([state, x, t], dim=-1)
        x = self.first_block(_x)
        residual = x

        for block in self.resnet:
            x = block(x)

        if resnet_skip:
            x = self.output_layer(x + residual)
        else:
            x = self.output_layer(x)

        return x


class MLPAdalNResNetBlock(nn.Module):
    """MLPResNet block."""
    def __init__(self, input_dim: int, hidden_dim: int, act: callable,
                 condition_dim: int = None,
                 dropout_rate: float = None, use_layer_norm: bool = False):
        super(MLPAdalNResNetBlock, self).__init__()
        self.act = act
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        # Define layers
        self.dense1 = nn.Linear(input_dim,  hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, input_dim)
        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_rate)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)

        self.adal_norm = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 3* input_dim),
        )

    def forward(self, x, c):
        residual = x
        gate, scale, shift = self.adal_norm(c).chunk(3, dim=-1)

        if self.dropout_rate is not None and self.dropout_rate > 0.0:
            x = self.dropout(x)
        if self.use_layer_norm:
            x = self.layer_norm(x)

        x = x * (1+ scale) + shift

        x = self.dense1(x)
        x = self.act(x)
        x = self.dense2(x)

        x = x * gate

        return residual + x


class MLPAdalNResNet(nn.Module):
    def __init__(self, state_dim, action_dim, device, num_blocks, hidden_dim, t_dim=64):
        super(MLPAdalNResNet, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.Mish(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # CONDITION CONTAINS REWARD AND RETURN
        self.r_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim*4),
            nn.Mish(),
            nn.Linear(hidden_dim*4, t_dim),
        )
        self.c_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim*4),
            nn.Mish(),
            nn.Linear(hidden_dim*4, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim

        self.first_block = nn.Linear(input_dim, hidden_dim)

        self.resnet = nn.ModuleList(
            [MLPAdalNResNetBlock(hidden_dim, hidden_dim*4, nn.Mish(), condition_dim=t_dim*2, dropout_rate=0.1, use_layer_norm=True)
             for _ in range(num_blocks)
             ])

        self.output_layer = nn.Sequential(
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out output layers:
        nn.init.constant_(self.output_layer[-1].weight, 0)
        nn.init.constant_(self.output_layer[-1].bias, 0)

    def forward(self, x, time, state, cond, resnet_skip=True):
        r_embedding = self.r_embedding(cond[:,0:1])
        c_embedding = self.c_embedding(cond[:,1:2])

        r_uncond_mask = cond[:, 2:3]
        c_uncond_mask = cond[:, 3:4]
        r_embedding = r_uncond_mask * r_embedding
        c_embedding = c_uncond_mask * c_embedding

        t = self.time_mlp(time)
        condition = torch.cat([r_embedding, c_embedding], dim=-1)
        _x = torch.cat([state, x, t], dim=-1)
        x = self.first_block(_x)
        residual = x

        for block in self.resnet:
            x = block(x, condition)

        if resnet_skip:
            x = self.output_layer(x + residual)
        else:
            x = self.output_layer(x)

        return x


class MLPVField(nn.Module):
    """MLP FOR FLOW MATCHING."""
    def __init__(self, state_dim, action_dim, device, hidden_dim, t_dim=64):
        super(MLPVField, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.GELU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # CONDITION CONTAINS REWARD AND RETURN
        self.r_embedding = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim*2

        self.net = mlp([input_dim] + [hidden_dim, hidden_dim] + [action_dim], nn.GELU, layernorm=True)

        self.residual = nn.Linear(input_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # # Zero-out output layers:
        # nn.init.constant_(self.output_layer[-1].weight, 0)
        # nn.init.constant_(self.output_layer[-1].bias, 0)

    def forward(self, x, time, state, cond, resnet_skip=True):
        r_embedding = self.r_embedding(cond)

        t = self.time_mlp(time)
        t = torch.cat([t, r_embedding], dim=-1)
        _x = torch.cat([state, x, t], dim=-1)
        residual = self.residual(_x)
        x = self.net(_x)

        if resnet_skip:
            return x + residual
        else:
            return x


class MLPVFieldUnCond(nn.Module):
    """MLP FOR FLOW MATCHING."""
    def __init__(self, state_dim, action_dim, device, hidden_dim, t_dim=64):
        super(MLPVFieldUnCond, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.GELU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim

        self.net = mlp([input_dim] + [hidden_dim, hidden_dim] + [action_dim], nn.GELU, layernorm=True)

        self.residual = nn.Linear(input_dim, action_dim)

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # # Zero-out output layers:
        # nn.init.constant_(self.output_layer[-1].weight, 0)
        # nn.init.constant_(self.output_layer[-1].bias, 0)

    def forward(self, x, time, state, resnet_skip=True):

        t = self.time_mlp(time)
        _x = torch.cat([state, x, t], dim=-1)
        residual = self.residual(_x)
        x = self.net(_x)

        if resnet_skip:
            return x + residual
        else:
            return x


class FISOR(object):
    def __init__(self, state_dim, action_dim, max_action, device, config, saving_logwriter=False):

        self.model = MLPVFieldUnCond(state_dim=state_dim, action_dim=action_dim,
                               device=device, hidden_dim=config['hidden_dim'])

        self.actor = DiffusionUnCond(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=config['beta_schedule'], n_timesteps=5,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_low = Policy(obs_dim=state_dim*2,
                                action_dim=action_dim,
                                hidden_dims=[512, 512],
                                ).to(device)
        self.actor_low_optimizer = torch.optim.Adam(self.actor_low.parameters(), lr=3e-4)
        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim, action_dim, [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
        else:
            raise NotImplementedError
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim,
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
        else:
            raise NotImplementedError
        self.cost_critic_target = copy.deepcopy(self.cost_critic)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim, [512, 512], nn.Mish, num_v=config['num_v']).to(device)
        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.Mish, num_v=config['num_vc']).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']


        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.lamb = 0.75

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_critic(self, state, next_state, action, reward, not_done):

        with torch.no_grad():
            qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(qc_list), dim=0).values, min=0.)
            vc = self.cost_value(state)[0]
            safe_mask = (qc-vc<0).float()

            next_value = self.value(next_state)[0]
            target = reward + not_done * self.discount * next_value

        _, qr_list = self.critic.predict(state, action)
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'q_loss': critic_loss,
                'batch_mean_q': target.mean(),
                'q_safe_rate': safe_mask.sum()/safe_mask.shape[0]}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        with torch.no_grad():
            next_value = self.cost_value(next_state)[0]

            qc_nonterminal = (1. - self.discount) * cost + self.discount * torch.maximum(
                cost, next_value)
            target_qc = qc_nonterminal * not_done + cost * (1 - not_done)
            # target_qc = self.cost_scale * cost + self.discount * not_done * next_value

        qc_list = self.cost_critic(state, action)
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target_qc)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target_qc,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target_qc.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):

        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value = self.value(state)[0]
        u = target_q - value
        expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
        value_loss = torch.mean(expectile_weight * u**2)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'v_loss': value_loss,
                'target_q_for_v_training': target_q,}

    def train_cost_value(self, state, action):

        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc = self.cost_value(state)[0]
        u = qc - vc
        cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
        vc_loss = torch.mean(cost_expectile_weight * u ** 2)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_actor(self, state, action):
        """
        NO GUIDANCE, ONLY EMIT ALL UNSAFE STATE ACTION PAIRS FROM TRAINING. WHEN TAKING ACTIONS, SELECT FROM CANDIDATES.
        """

        with torch.no_grad():
            eps = 0.
            qc_list = self.cost_critic_target(state, action)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc = self.cost_value(state)[0]
            cost_adv = (qc - vc)/torch.abs(qc-vc).max()

            q_list = self.critic_target(state, action)
            q = torch.min(torch.stack(q_list), dim=0).values
            v = self.value(state)[0]

            unsafe_condition = torch.where(vc > 0. - eps, torch.tensor(1.0), torch.tensor(0.0))
            safe_condition = torch.where(vc <= 0. - eps, torch.tensor(1.0), torch.tensor(0.0)) * \
                             torch.where(qc <= 0. - eps, torch.tensor(1.0), torch.tensor(0.0))

            cost_exp_adv = torch.exp((vc - qc) * 3.0)
            reward_exp_adv = torch.exp((q - v) * 3.0)

            unsafe_weights = unsafe_condition * torch.clamp(cost_exp_adv, max=150.0)
            safe_weights = safe_condition * torch.clamp(reward_exp_adv, max=100.0)

            weights = unsafe_weights + safe_weights

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        actor_loss = self.actor.loss(action, state, weights)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'cost_adv_for_actor_training': cost_adv,
                'Actor Loss': actor_loss,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        metrics = {}
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample(batch_size=batch_size*4)

        """ POLICY LEARNING """
        metrics.update(self.train_actor(state, action))

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()

        return metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        (state, next_state, action, next_action, reward, cost,
         not_done) = replay_buffer.sample(batch_size, sample_setting='next_action')
        reward = self.reward_scale * reward

        cost = torch.where(cost > 0, torch.tensor(25., device=cost.device, dtype=cost.dtype),
                           torch.tensor(-1., device=cost.device, dtype=cost.dtype))

        """ VALUE TRAINING """
        metrics.update(self.train_value(state, action))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, action, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state, action))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, action, cost, not_done))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def select_action_from_candidates(self, state, eval=True):
        self.ema_model.eval()

        if not torch.is_tensor(state):

            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=16, dim=0) # [100, STATE_DIM]

        if eval:
            with torch.no_grad():
                action = self.ema_model.sample(state_rpt) # [100, ACTION_DIM]

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.mean(torch.stack(qc_list), dim=0)

                idx = torch.argmin(qc_mean)
        else:
            action = self.ema_model.sample(state_rpt)

            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor

    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),

                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)

    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list


class FISORV1(object):
    def __init__(self, state_dim, action_dim, max_action, device, config, saving_logwriter=False):

        self.model = MLPResNetV1(state_dim=state_dim, action_dim=state_dim,
                               device=device, num_blocks=config['actor_num_stack'], hidden_dim=config['hidden_dim'])

        self.actor = DiffusionV1(state_dim=state_dim, action_dim=state_dim, model=self.model, max_action=max_action,
                               beta_schedule=config['beta_schedule'], n_timesteps=config['T'],).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_low = Policy(obs_dim=state_dim*2,
                                action_dim=action_dim,
                                hidden_dims=[512, 512],
                                ).to(device)
        self.actor_low_optimizer = torch.optim.Adam(self.actor_low.parameters(), lr=3e-4)
        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim, action_dim, [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
        else:
            raise NotImplementedError
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim,
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
        else:
            raise NotImplementedError
        self.cost_critic_target = copy.deepcopy(self.cost_critic)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim, [512, 512], nn.Mish, num_v=config['num_v']).to(device)
        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.Mish, num_v=config['num_vc']).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']

        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.lamb = 0.75

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_critic(self, state, next_state, action, reward, not_done):

        with torch.no_grad():
            qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(qc_list), dim=0).values, min=0.)
            vc_list = self.cost_value(state)
            vc = torch.clamp(torch.max(torch.stack(vc_list), dim=0).values, min=0.)
            safe_mask = (qc<0.1).float()

            next_value_list = self.value(next_state)
            next_value_ = torch.min(torch.stack(next_value_list), dim=0).values
            target = reward + not_done * self.discount * next_value_

        _, qr_list = self.critic.predict(state, action)
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'batch_mean_q': target.mean(),
                'next_value_for_q_training': next_value_,
                'q_loss': critic_loss,
                'q_safe_rate': safe_mask.sum()/safe_mask.shape[0]}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        with torch.no_grad():
            next_value_list = self.cost_value(next_state)
            next_value = torch.clamp(torch.max(torch.stack(next_value_list), dim=0).values, min=0)

            target_qc = cost + self.discount * not_done * next_value

        qc_list = self.cost_critic(state, action)
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target_qc)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target_qc,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target_qc.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):

        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value_list = self.value(state)
        value_loss_list = []
        for value in value_list:
            u = target_q - value
            expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
            expectile_loss = torch.mean(expectile_weight * u**2)
            value_loss_list.append(expectile_loss)

        value_loss = sum(value_loss_list)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'target_q_for_v_training': target_q,
                'batch_v_in_train_value': torch.max(torch.stack(value_list), dim=0).values,
                'v_loss': value_loss,}

    def train_cost_value(self, state, action):

        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc_list = self.cost_value(state)
        vc_loss_list = []
        for vc in vc_list:
            u = qc - vc
            cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
            expectile_loss = torch.mean(cost_expectile_weight * u ** 2)
            vc_loss_list.append(expectile_loss)

        vc_loss = sum(vc_loss_list)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_actor(self, state, action):
        """
        NO GUIDANCE, ONLY EMIT ALL UNSAFE STATE ACTION PAIRS FROM TRAINING. WHEN TAKING ACTIONS, SELECT FROM CANDIDATES.
        """

        with torch.no_grad():

            qc_list = self.cost_critic_target(state, action)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_list = self.cost_value(state)
            vc = torch.max(torch.stack(vc_list), dim=0).values
            cadv = (qc - vc)
            cost_adv = (qc - vc)/torch.abs(qc-vc).max()

            q_list = self.critic_target(state, action)
            q = torch.min(torch.stack(q_list), dim=0).values
            v_list = self.value(state)
            v = torch.mean(torch.stack(v_list), dim=0)
            adv = (q-v)/torch.abs(q-v).max()

            condition = torch.cat([adv, cost_adv], dim=-1)
            condition = torch.cat([condition, torch.ones_like(condition[:,:2])], dim=-1)
            p = 0.5
            mask1 = torch.rand(condition.shape[0]) < p
            mask2 = torch.rand(condition.shape[0]) < p
            condition[:,2][mask1] = 0.
            condition[:,3][mask2] = 0.

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        actor_loss = self.actor.loss(action, state, condition)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'cost_adv_for_actor_training': cost_adv,
                'adv_for_actor_training': adv,
                'Actor Loss': actor_loss,}

    def train_high_actor(self, state):
        with torch.no_grad():

            vc_first_list = self.cost_value(state[:,0])
            vc_last_list = self.cost_value(state[:, -1])
            vc_first = torch.max(torch.stack(vc_first_list), dim=0).values
            vc_last = torch.max(torch.stack(vc_last_list), dim=0).values
            cost_adv = (vc_last-vc_first)

            vr_first_list = self.value(state[:, 0])
            vr_last_list = self.value(state[:, -1])
            vr_first = torch.max(torch.stack(vr_first_list), dim=0).values
            vr_last = torch.max(torch.stack(vr_last_list), dim=0).values
            adv = (vr_last - vr_first)

            condition = torch.cat([adv, cost_adv], dim=-1)
            condition = torch.zeros_like(condition)
            mask1 = adv.squeeze() > 0
            mask2 = cost_adv.squeeze() < 0
            condition[:,0][mask1] = 1.
            condition[:,1][mask2] = 1.
            p = 0.1
            mask1 = torch.rand(condition.shape[0]) < p
            mask2 = torch.rand(condition.shape[0]) < p
            condition[:,0][mask1] = 0.
            condition[:,1][mask2] = 0.

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        actor_loss = self.actor.loss(state[:, -1], state[:, 0], condition)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'cost_adv_for_high_actor_training': cost_adv,
                'adv_for_high_actor_training': adv,
                'High Actor Loss': actor_loss,}


    def train_low_actor(self, state, action):

        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:,0], action[:,0])
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_list = self.cost_value(state[:,0])
            vc = torch.max(torch.stack(vc_list), dim=0).values
            cost_adv = (qc - vc)
            exp_b = torch.clamp(torch.exp(-cost_adv*self.cost_temperature), min=0.0, max=200.0)

            q_list = self.critic_target(state[:,0], action[:,0])
            q = torch.min(torch.stack(q_list), dim=0).values
            v_list = self.value(state[:,0])
            v = torch.mean(torch.stack(v_list), dim=0)
            adv = (q - v)
            exp_a = torch.exp(adv * self.reward_temperature)
            exp_a = torch.clamp(exp_a, max=100.0)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        random_idx = torch.randint(1, state.shape[1], (state.shape[0],))
        inputs = torch.cat([state[:, 0], state[torch.arange(state.shape[0]), random_idx]], dim=-1)

        dist = self.actor_low(inputs)
        log_probs = dist.log_prob(action[:,0]).sum(-1)
        actor_loss_low = - (exp_b * log_probs).mean()

        self.actor_low_optimizer.zero_grad()
        actor_loss_low.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor_low.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_low_optimizer.step()

        return {'cost_adv_for_low_actor_training': cost_adv,
                'adv_for_low_actor_training': adv,
                'Low Actor Loss': actor_loss_low,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample_k_step_trajectory(k=10, batch_size=batch_size*4)

        """ POLICY LEARNING """
        actor_high_metrics = self.train_high_actor(state)
        actor_low_metrics = self.train_low_actor(state, action)
        actor_high_metrics.update(actor_low_metrics)

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()

        return actor_high_metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        (state, next_state, action, next_action, reward, cost,
         not_done) = replay_buffer.sample(batch_size, sample_setting='next_action')
        reward = self.reward_scale * reward

        """ VALUE TRAINING """
        metrics.update(self.train_value(state, action))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, action, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state, action))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, action, cost, not_done))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def select_action_from_candidates(self, state, eval=True):

        self.ema_model.eval()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=16, dim=0) # [100, STATE_DIM]
        cond = torch.ones((state_rpt.shape[0], 2), device=state.device)
        if eval:
            with torch.no_grad():
                sk = self.ema_model.sample(state_rpt, cond) # [100, ACTION_DIM]
                inputs = torch.cat([state_rpt, sk], dim=1)
                dist = self.actor_low(inputs)
                action = dist.rsample()

                q_list = self.critic_target(state_rpt, action)
                q_mean = torch.mean(torch.stack(q_list), dim=0)

                v_list = self.value(state_rpt)
                v_mean = torch.mean(torch.stack(v_list), dim=0)
                adv = q_mean - v_mean

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.mean(torch.stack(qc_list), dim=0)

                vc_list = self.cost_value(state_rpt)
                vc_mean = torch.mean(torch.stack(vc_list), dim=0)

                cost_adv = qc_mean - vc_mean #+ self.qc_penalized_coef * torch.std(qc_std**2 + vc_std**2)

                idx = torch.argmin(cost_adv)
        else:
            action = self.ema_model.sample(state_rpt)
            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor


    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)


    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list


class FISORV2(object):
    def __init__(self, state_dim, action_dim, max_action, device, config, saving_logwriter=False):

        self.model = MLPResNetV2(state_dim=state_dim, action_dim=state_dim,
                               device=device, num_blocks=config['actor_num_stack'], hidden_dim=config['hidden_dim'])

        self.actor = DiffusionV2(state_dim=state_dim, action_dim=state_dim, model=self.model, max_action=max_action,
                               beta_schedule=config['beta_schedule'], n_timesteps=config['T'],).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_low = Policy(obs_dim=state_dim*2,
                                action_dim=action_dim,
                                hidden_dims=[512, 512],
                                ).to(device)
        self.actor_low_optimizer = torch.optim.Adam(self.actor_low.parameters(), lr=3e-4)
        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim, action_dim, [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
        else:
            raise NotImplementedError
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim,
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
        else:
            raise NotImplementedError
        self.cost_critic_target = copy.deepcopy(self.cost_critic)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim, [512, 512], nn.Mish, num_v=config['num_v']).to(device)
        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.Mish, num_v=config['num_vc']).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']

        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.lamb = 0.75

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_critic(self, state, next_state, action, reward, not_done):

        with torch.no_grad():
            qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(qc_list), dim=0).values, min=0.)
            vc_list = self.cost_value(state)
            vc = torch.clamp(torch.max(torch.stack(vc_list), dim=0).values, min=0.)
            safe_mask = (qc<0.1).float()

            next_value_list = self.value(next_state)
            next_value_ = torch.min(torch.stack(next_value_list), dim=0).values
            target = reward + not_done * self.discount * next_value_

        _, qr_list = self.critic.predict(state, action)
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'batch_mean_q': target.mean(),
                'next_value_for_q_training': next_value_,
                'q_loss': critic_loss,
                'q_safe_rate': safe_mask.sum()/safe_mask.shape[0]}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        with torch.no_grad():
            next_value_list = self.cost_value(next_state)
            next_value = torch.clamp(torch.max(torch.stack(next_value_list), dim=0).values, min=0)

            target_qc = cost + self.discount * not_done * next_value

        qc_list = self.cost_critic(state, action)
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target_qc)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target_qc,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target_qc.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):

        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value_list = self.value(state)
        value_loss_list = []
        for value in value_list:
            u = target_q - value
            expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
            expectile_loss = torch.mean(expectile_weight * u**2)
            value_loss_list.append(expectile_loss)

        value_loss = sum(value_loss_list)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'target_q_for_v_training': target_q,
                'batch_v_in_train_value': torch.max(torch.stack(value_list), dim=0).values,
                'v_loss': value_loss,}

    def train_cost_value(self, state, action):

        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc_list = self.cost_value(state)
        vc_loss_list = []
        for vc in vc_list:
            u = qc - vc
            cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
            expectile_loss = torch.mean(cost_expectile_weight * u ** 2)
            vc_loss_list.append(expectile_loss)

        vc_loss = sum(vc_loss_list)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_high_actor(self, state):
        with torch.no_grad():

            vc_first_list = self.cost_value(state[:,0])
            vc_last_list = self.cost_value(state[:, -1])
            vc_first = torch.max(torch.stack(vc_first_list), dim=0).values
            vc_last = torch.max(torch.stack(vc_last_list), dim=0).values
            cost_adv = (vc_last-vc_first)

            vr_first_list = self.value(state[:, 0])
            vr_last_list = self.value(state[:, -1])
            vr_first = torch.max(torch.stack(vr_first_list), dim=0).values
            vr_last = torch.max(torch.stack(vr_last_list), dim=0).values
            adv = (vr_last - vr_first)

            condition_0 = torch.zeros_like(adv).squeeze()
            condition_1 = torch.ones_like(adv).squeeze()
            mask = (adv > 0).float()

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        guided_loss = self.actor.loss(state[:, -1], state[:, 0], condition_1, weights=mask)
        unguided_loss = self.actor.loss(state[:, -1], state[:, 0], condition_0)
        actor_loss = 0.5 * guided_loss + 0.5 * unguided_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'cost_adv_for_high_actor_training': cost_adv,
                'adv_for_high_actor_training': adv,
                'High Actor Loss': actor_loss,}


    def train_low_actor(self, state, action):

        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:,0], action[:,0])
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_list = self.cost_value(state[:,0])
            vc = torch.max(torch.stack(vc_list), dim=0).values
            cost_adv = (qc - vc)
            exp_b = torch.clamp(torch.exp(-cost_adv*self.cost_temperature), min=0.0, max=200.0)

            q_list = self.critic_target(state[:,0], action[:,0])
            q = torch.min(torch.stack(q_list), dim=0).values
            v_list = self.value(state[:,0])
            v = torch.mean(torch.stack(v_list), dim=0)
            adv = (q - v)
            exp_a = torch.exp(adv * self.reward_temperature)
            exp_a = torch.clamp(exp_a, max=100.0)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        inputs = torch.cat([state[:, 0], state[:, -1]], dim=-1)
        dist = self.actor_low(inputs)
        log_probs = dist.log_prob(action[:,0]).sum(-1)
        actor_low_minc_loss = - (exp_b * log_probs).mean()

        # random_idx = torch.randint(1, state.shape[1], (state.shape[0],))
        # inputs = torch.cat([state[:, 0], state[torch.arange(state.shape[0]), random_idx]], dim=-1)
        # dist = self.actor_low(inputs)
        # log_probs = dist.log_prob(action[:,0]).sum(-1)
        # actor_low_reaching_loss = - (log_probs).mean()

        actor_low_loss = 0.5 * actor_low_minc_loss# + 0.5 * actor_low_reaching_loss

        self.actor_low_optimizer.zero_grad()
        actor_low_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor_low.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_low_optimizer.step()

        return {'cost_adv_for_low_actor_training': cost_adv,
                'adv_for_low_actor_training': adv,
                'Low Actor Loss': actor_low_loss,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample_k_step_trajectory(k=2, batch_size=batch_size*4)

        """ POLICY LEARNING """
        actor_high_metrics = self.train_high_actor(state)
        actor_low_metrics = self.train_low_actor(state, action)
        actor_high_metrics.update(actor_low_metrics)

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()

        return actor_high_metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        (state, next_state, action, next_action, reward, cost,
         not_done) = replay_buffer.sample(batch_size, sample_setting='next_action')
        reward = self.reward_scale * reward

        """ VALUE TRAINING """
        metrics.update(self.train_value(state, action))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, action, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state, action))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, action, cost, not_done))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def select_action_from_candidates(self, state, eval=True):

        self.ema_model.eval()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=16, dim=0) # [100, STATE_DIM]
        cond = torch.ones((state_rpt.shape[0],), device=state.device)
        if eval:
            with torch.no_grad():
                sk = self.ema_model.sample(state_rpt, cond) # [100, ACTION_DIM]
                inputs = torch.cat([state_rpt, sk], dim=1)
                dist = self.actor_low(inputs)
                action = dist.rsample()

                q_list = self.critic_target(state_rpt, action)
                q_mean = torch.mean(torch.stack(q_list), dim=0)

                v_list = self.value(state_rpt)
                v_mean = torch.mean(torch.stack(v_list), dim=0)
                adv = q_mean - v_mean

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.mean(torch.stack(qc_list), dim=0)

                vc_list = self.cost_value(state_rpt)
                vc_mean = torch.mean(torch.stack(vc_list), dim=0)

                cost_adv = qc_mean - vc_mean #+ self.qc_penalized_coef * torch.std(qc_std**2 + vc_std**2)

                idx = torch.argmin(cost_adv)
        else:
            action = self.ema_model.sample(state_rpt)
            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor


    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)


    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list


class FISORV3(object):
    """
    This version uses n-step return for value function update.
    All are same with V2/
    """
    def __init__(self, state_dim, action_dim, max_action, device, config, saving_logwriter=False):

        # self.model = MLPResNetV2(state_dim=state_dim, action_dim=state_dim,
        #                        device=device, num_blocks=config['actor_num_stack'], hidden_dim=config['hidden_dim'])
        self.model = MLPVField(state_dim=state_dim, action_dim=state_dim,
                               device=device, hidden_dim=config['hidden_dim'])

        self.actor = DiffusionV2(state_dim=state_dim, action_dim=state_dim, model=self.model, max_action=max_action,
                               beta_schedule=config['beta_schedule'], n_timesteps=config['T'],).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.actor_low = Policy(obs_dim=state_dim*2,
                                action_dim=action_dim,
                                hidden_dims=[512, 512],
                                ).to(device)
        self.actor_low_optimizer = torch.optim.Adam(self.actor_low.parameters(), lr=3e-4)
        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim, action_dim, [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
        else:
            raise NotImplementedError
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim,
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
        else:
            raise NotImplementedError
        self.cost_critic_target = copy.deepcopy(self.cost_critic)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim, [512, 512], nn.Mish, num_v=config['num_v']).to(device)
        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.Mish, num_v=config['num_vc']).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']

        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.k = config['guided_step']
        self.chunking = config['chunking_length']

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_critic(self, state, next_state, action, reward, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(reward[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount**i) * reward[:, i]

            next_value_list = self.value(next_state[:, -1])
            next_value_ = torch.min(torch.stack(next_value_list), dim=0).values
            target = traj_return + not_done[:, -1] * (self.discount**seq_len) * next_value_

        _, qr_list = self.critic.predict(state[:, 0], action[:, 0])
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'batch_mean_q': target.mean(),
                'next_value_for_q_training': next_value_,
                'q_loss': critic_loss,}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(cost[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount ** i) * cost[:, i]

            next_value_list = self.cost_value(next_state[:, -1])
            next_value = torch.clamp(torch.max(torch.stack(next_value_list), dim=0).values, min=0)

            target = traj_return + not_done[:, -1] * (self.discount ** seq_len) * next_value

        qc_list = self.cost_critic(state[:,0], action[:,0])
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):

        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value_list = self.value(state)
        value_loss_list = []
        for value in value_list:
            u = target_q - value
            expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
            expectile_loss = torch.mean(expectile_weight * u**2)
            value_loss_list.append(expectile_loss)

        value_loss = sum(value_loss_list)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'target_q_for_v_training': target_q,
                'batch_v_in_train_value': torch.max(torch.stack(value_list), dim=0).values,
                'v_loss': value_loss,}

    def train_cost_value(self, state, action):

        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc_list = self.cost_value(state)
        vc_loss_list = []
        for vc in vc_list:
            u = qc - vc
            cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
            expectile_loss = torch.mean(cost_expectile_weight * u ** 2)
            vc_loss_list.append(expectile_loss)

        vc_loss = sum(vc_loss_list)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_high_actor(self, state):
        with torch.no_grad():

            vc_first_list = self.cost_value(state[:,0])
            vc_last_list = self.cost_value(state[:, -1])
            vc_first = torch.max(torch.stack(vc_first_list), dim=0).values
            vc_last = torch.max(torch.stack(vc_last_list), dim=0).values
            cost_adv = (vc_last-vc_first)

            vr_first_list = self.value(state[:, 0])
            vr_last_list = self.value(state[:, -1])
            vr_first = torch.max(torch.stack(vr_first_list), dim=0).values
            vr_last = torch.max(torch.stack(vr_last_list), dim=0).values
            adv = (vr_last - vr_first)

            condition_0 = torch.zeros_like(adv).squeeze()
            condition_1 = torch.ones_like(adv).squeeze()
            mask = (adv > 0).float()

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        guided_loss = self.actor.loss(state[:, -1], state[:, 0], condition_1, weights=mask)
        unguided_loss = self.actor.loss(state[:, -1], state[:, 0], condition_0)
        actor_loss = 0.5 * guided_loss + 0.5 * unguided_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'cost_adv_for_high_actor_training': cost_adv,
                'adv_for_high_actor_training': adv,
                'High Actor Loss': actor_loss,}


    def train_low_actor(self, state, action):

        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:,0], action[:,0])
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_list = self.cost_value(state[:,0])
            vc = torch.max(torch.stack(vc_list), dim=0).values
            cost_adv = (qc - vc)
            exp_b = torch.clamp(torch.exp(-cost_adv*self.cost_temperature), min=0.0, max=200.0)

            q_list = self.critic_target(state[:,0], action[:,0])
            q = torch.min(torch.stack(q_list), dim=0).values
            v_list = self.value(state[:,0])
            v = torch.mean(torch.stack(v_list), dim=0)
            adv = (q - v)
            exp_a = torch.exp(adv * self.reward_temperature)
            exp_a = torch.clamp(exp_a, max=100.0)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        inputs = torch.cat([state[:, 0], state[:, -1]], dim=-1)
        dist = self.actor_low(inputs)
        log_probs = dist.log_prob(action[:,0]).sum(-1)
        actor_low_minc_loss = - (exp_b * log_probs).mean()
        actor_low_loss = 0.5 * actor_low_minc_loss

        self.actor_low_optimizer.zero_grad()
        actor_low_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor_low.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_low_optimizer.step()

        return {'cost_adv_for_low_actor_training': cost_adv,
                'adv_for_low_actor_training': adv,
                'Low Actor Loss': actor_low_loss,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample_k_step_trajectory(k=self.k, batch_size=batch_size*4)

        """ POLICY LEARNING """
        actor_high_metrics = self.train_high_actor(state)
        actor_low_metrics = self.train_low_actor(state, action)
        actor_high_metrics.update(actor_low_metrics)

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()

        return actor_high_metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        state, next_state, action, reward, cost, not_done =\
            replay_buffer.sample_k_step_trajectory(k=self.chunking, batch_size=batch_size)
        reward = self.reward_scale * reward

        """ VALUE TRAINING """
        metrics.update(self.train_value(state[:,0], action[:,0]))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, action, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state[:, 0], action[:, 0]))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, action, cost, not_done))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def select_action_from_candidates(self, state, eval=True):

        self.ema_model.eval()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=16, dim=0) # [100, STATE_DIM]
        cond = torch.ones((state_rpt.shape[0],), device=state.device)
        if eval:
            with torch.no_grad():
                sk = self.ema_model.sample(state_rpt, cond) # [100, ACTION_DIM]
                inputs = torch.cat([state_rpt, sk], dim=1)
                dist = self.actor_low(inputs)
                action = dist.rsample()

                q_list = self.critic_target(state_rpt, action)
                q_mean = torch.mean(torch.stack(q_list), dim=0)

                v_list = self.value(state_rpt)
                v_mean = torch.mean(torch.stack(v_list), dim=0)
                adv = q_mean - v_mean

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.mean(torch.stack(qc_list), dim=0)

                vc_list = self.cost_value(state_rpt)
                vc_mean = torch.mean(torch.stack(vc_list), dim=0)

                cost_adv = qc_mean - vc_mean #+ self.qc_penalized_coef * torch.std(qc_std**2 + vc_std**2)

                idx = torch.argmin(cost_adv)
        else:
            action = self.ema_model.sample(state_rpt)
            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor


    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)


    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list


class FLOWCHUNK(object):
    """
    Action chunking + Diffusion + Two-level policy structure
    """
    def __init__(self, state_dim, action_dim, max_action, device, config, saving_logwriter=False):

        self.model = MLPVField(state_dim=state_dim, action_dim=state_dim,
                               device=device, hidden_dim=config['hidden_dim'])

        self.actor = FlowMatching(state_dim=state_dim,
                                  action_dim=state_dim,
                                  model=self.model,
                                  max_action=torch.inf,
                                  denoise_steps=16,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.model_low = MLPVField(state_dim=state_dim*2,
                                     action_dim=action_dim*config['chunking_length'],
                                     device=device,
                                     hidden_dim=config['hidden_dim'])
        self.actor_low = FlowMatching(state_dim=state_dim*2,
                                  action_dim=action_dim*config['chunking_length'],
                                  model=self.model_low,
                                  max_action=max_action,
                                  denoise_steps=16,).to(device)
        self.actor_low_optimizer = torch.optim.Adam(self.actor_low.parameters(), lr=3e-4)
        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim,
                                          action_dim*config['chunking_length'],
                                          [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim*config['chunking_length'],
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        else:
            raise NotImplementedError

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim,
                                       [512, 512],
                                       nn.Mish, num_v=config['num_v']).to(device)
        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.Mish, num_v=config['num_vc']).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']

        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.k = config['guided_step']
        self.chunking = config['chunking_length']

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_critic(self, state, next_state, action, reward, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(reward[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount**i) * reward[:, i]

            next_value_list = self.value(next_state[:, -1])
            next_value_ = torch.min(torch.stack(next_value_list), dim=0).values
            target = traj_return + not_done[:, -1] * (self.discount**seq_len) * next_value_

        qr_list = self.critic(state[:, 0], action)
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'batch_mean_q': target.mean(),
                'next_value_for_q_training': next_value_,
                'q_loss': critic_loss,}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(cost[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount ** i) * cost[:, i]

            next_value_list = self.cost_value(next_state[:, -1])
            next_value = torch.clamp(torch.max(torch.stack(next_value_list), dim=0).values, min=0)

            target = traj_return + not_done[:, -1] * (self.discount ** seq_len) * next_value

        qc_list = self.cost_critic(state[:,0], action)
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):
        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value_list = self.value(state)
        value_loss_list = []
        for value in value_list:
            u = target_q - value
            expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
            expectile_loss = torch.mean(expectile_weight * u**2)
            value_loss_list.append(expectile_loss)

        value_loss = sum(value_loss_list)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'target_q_for_v_training': target_q,
                'batch_v_in_train_value': torch.max(torch.stack(value_list), dim=0).values,
                'v_loss': value_loss,}

    def train_cost_value(self, state, action):
        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc_list = self.cost_value(state)
        vc_loss_list = []
        for vc in vc_list:
            u = qc - vc
            cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
            expectile_loss = torch.mean(cost_expectile_weight * u ** 2)
            vc_loss_list.append(expectile_loss)

        vc_loss = sum(vc_loss_list)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_high_actor(self, state):
        with torch.no_grad():

            vc_first_list = self.cost_value(state[:,0])
            vc_last_list = self.cost_value(state[:, -1])
            vc_first = torch.max(torch.stack(vc_first_list), dim=0).values
            vc_last = torch.max(torch.stack(vc_last_list), dim=0).values
            cost_adv = (vc_last-vc_first)

            vr_first_list = self.value(state[:, 0])
            vr_last_list = self.value(state[:, -1])
            vr_first = torch.max(torch.stack(vr_first_list), dim=0).values
            vr_last = torch.max(torch.stack(vr_last_list), dim=0).values
            adv = (vr_last - vr_first)

            condition_0 = torch.zeros_like(adv).squeeze()
            condition_1 = torch.ones_like(adv).squeeze()
            mask = ((adv-cost_adv) > 0).float()

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        guided_loss = self.actor.loss(state[:, -1], state[:, 0], condition_1, weights=mask)
        unguided_loss = self.actor.loss(state[:, -1], state[:, 0], condition_0)
        actor_loss = guided_loss + 0.1 * unguided_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'cost_adv_for_high_actor_training': cost_adv,
                'adv_for_high_actor_training': adv,
                'High Actor Loss': actor_loss,
                'high_level_guided_loss': guided_loss,
                'high_level_unguided_loss': unguided_loss,}


    def train_low_actor(self, state, action):
        bs = state.shape[0]
        seq_a = action[:, :self.chunking].reshape(bs, -1)
        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:,0], seq_a)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_list = self.cost_value(state[:,0])
            vc = torch.max(torch.stack(vc_list), dim=0).values
            cost_adv = (qc - vc)
            exp_b = torch.clamp(torch.exp(-cost_adv*self.cost_temperature), min=0.0, max=200.0)

            q_list = self.critic_target(state[:,0], seq_a)
            q = torch.min(torch.stack(q_list), dim=0).values
            v_list = self.value(state[:,0])
            v = torch.mean(torch.stack(v_list), dim=0)
            adv = (q - v)
            exp_a = torch.exp(adv * self.reward_temperature)
            exp_a = torch.clamp(exp_a, max=100.0)

            cond_0 = torch.zeros_like(adv).squeeze()
            cond_1 = torch.ones_like(adv).squeeze()
            mask = (cost_adv < 0).float()
            inputs = torch.cat([state[:, 0], state[:, -1]], dim=-1)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        guided_loss = self.actor_low.loss(seq_a, inputs, cond_1, weights=mask)
        unguided_loss = self.actor_low.loss(seq_a, inputs, cond_0)
        actor_low_loss = guided_loss + 0.1 * unguided_loss

        self.actor_low_optimizer.zero_grad()
        actor_low_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor_low.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_low_optimizer.step()

        return {'cost_adv_for_low_actor_training': cost_adv,
                'adv_for_low_actor_training': adv,
                'Low Actor Loss': actor_low_loss,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample_k_step_trajectory(k=self.k, batch_size=batch_size*4)

        """ POLICY LEARNING """
        actor_high_metrics = self.train_high_actor(state)
        actor_low_metrics = self.train_low_actor(state, action)
        actor_high_metrics.update(actor_low_metrics)

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()

        return actor_high_metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        state, next_state, action, reward, cost, not_done =\
            replay_buffer.sample_k_step_trajectory(k=self.chunking, batch_size=batch_size)
        reward = self.reward_scale * reward
        bs, seq_len, d = action.shape
        seq_a = action.reshape(bs, -1)

        """ VALUE TRAINING """
        metrics.update(self.train_value(state[:,0], seq_a))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, seq_a, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state[:, 0], seq_a))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, seq_a, cost, not_done))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def sample_sk(self, state):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)
        cond = torch.ones((state.shape[0],), device=state.device)
        with torch.no_grad():
            sk = self.ema_model.sample(state, cond, guidance_scale=3.0) # [100, ACTION_DIM]
        return sk.cpu().data.numpy()

    def sample_action(self, state):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)
        cond = torch.ones((state.shape[0],), device=state.device)
        with torch.no_grad():
            sk = self.ema_model.sample(state, cond, guidance_scale=3.0)
            inputs = torch.cat([state, sk], dim=1)
            action = self.actor_low.sample(inputs, cond, guidance_scale=3.0)
        return action[:,:self.action_dim].cpu().data.numpy()

    def select_action_from_candidates(self, state, eval=True):

        self.ema_model.eval()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=16, dim=0) # [100, STATE_DIM]
        cond = torch.ones((state_rpt.shape[0],), device=state.device)
        if eval:
            with torch.no_grad():
                sk = self.ema_model.sample(state_rpt, cond, guidance_scale=2.0) # [100, ACTION_DIM]
                inputs = torch.cat([state_rpt, sk], dim=1)
                action = self.actor_low.sample(inputs, cond, guidance_scale=2.0)

                q_list = self.critic_target(state_rpt, action)
                q_mean = torch.min(torch.stack(q_list), dim=0).values

                v_list = self.value(state_rpt)
                v_mean = torch.mean(torch.stack(v_list), dim=0)
                adv = q_mean - v_mean

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.max(torch.stack(qc_list), dim=0).values

                vc_list = self.cost_value(state_rpt)
                vc_mean = torch.mean(torch.stack(vc_list), dim=0)

                cost_adv = qc_mean - vc_mean #+ self.qc_penalized_coef * torch.std(qc_std**2 + vc_std**2)

                idx = torch.argmin(cost_adv)
        else:
            action = self.ema_model.sample(state_rpt)
            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            action = action[:, :self.action_dim]
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor


    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)


    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list


class FLOWCHUNKV1(object):
    """
    Action chunking + Diffusion + NO HIERARCHICAl
    """
    def __init__(self, state_dim, action_dim, max_action, device, config, saving_logwriter=False):

        self.model = MLPVField(state_dim=state_dim, action_dim=action_dim*config['chunking_length'],
                               device=device, hidden_dim=config['hidden_dim'])

        self.actor = FlowMatching(state_dim=state_dim,
                                  action_dim=action_dim*config['chunking_length'],
                                  model=self.model,
                                  max_action=max_action,
                                  denoise_steps=16,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim,
                                          action_dim*config['chunking_length'],
                                          [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim*config['chunking_length'],
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        else:
            raise NotImplementedError

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim,
                                       [512, 512],
                                       nn.Mish, num_v=config['num_v']).to(device)
        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.Mish, num_v=config['num_vc']).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']

        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.k = config['guided_step']
        self.chunking = config['chunking_length']

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_critic(self, state, next_state, action, reward, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(reward[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount**i) * reward[:, i]

            next_value_list = self.value(next_state[:, -1])
            next_value_ = torch.min(torch.stack(next_value_list), dim=0).values
            target = traj_return + not_done[:, -1] * (self.discount**seq_len) * next_value_

        qr_list = self.critic(state[:, 0], action)
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'batch_mean_q': target.mean(),
                'next_value_for_q_training': next_value_,
                'q_loss': critic_loss,}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(cost[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount ** i) * cost[:, i]

            next_value_list = self.cost_value(next_state[:, -1])
            next_value = torch.clamp(torch.max(torch.stack(next_value_list), dim=0).values, min=0)

            target = traj_return + not_done[:, -1] * (self.discount ** seq_len) * next_value

        qc_list = self.cost_critic(state[:,0], action)
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):
        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value_list = self.value(state)
        value_loss_list = []
        for value in value_list:
            u = target_q - value
            expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
            expectile_loss = torch.mean(expectile_weight * u**2)
            value_loss_list.append(expectile_loss)

        value_loss = sum(value_loss_list)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'target_q_for_v_training': target_q,
                'batch_v_in_train_value': torch.max(torch.stack(value_list), dim=0).values,
                'v_loss': value_loss,}

    def train_cost_value(self, state, action):
        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc_list = self.cost_value(state)
        vc_loss_list = []
        for vc in vc_list:
            u = qc - vc
            cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
            expectile_loss = torch.mean(cost_expectile_weight * u ** 2)
            vc_loss_list.append(expectile_loss)

        vc_loss = sum(vc_loss_list)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_high_actor(self, state, action):
        bs = state.shape[0]
        seq_a = action[:, :self.chunking].reshape(bs, -1)
        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:,0], seq_a)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_first_list = self.cost_value(state[:,0])
            vc_first = torch.max(torch.stack(vc_first_list), dim=0).values
            cost_adv = (qc-vc_first)

            q_list = self.critic_target(state[:, 0], seq_a)
            q = torch.min(torch.stack(q_list), dim=0).values
            vr_first_list = self.value(state[:, 0])
            vr_first = torch.min(torch.stack(vr_first_list), dim=0).values
            adv = (q - vr_first)

            condition_0 = torch.zeros_like(adv).squeeze()
            condition_1 = torch.ones_like(adv).squeeze()
            condition_m1 = - torch.ones_like(adv).squeeze()
            mask1 = (cost_adv < 0).float()
            mask2 = (adv > 0).float()

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        c_guided_loss = self.actor.loss(seq_a, state[:, 0], condition_1, weights=mask1)
        r_guided_loss = self.actor.loss(seq_a, state[:, 0], condition_m1, weights=mask2)
        unguided_loss = self.actor.loss(seq_a, state[:, 0], condition_0)
        actor_loss = r_guided_loss + c_guided_loss + 0.1 * unguided_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'cost_adv_for_high_actor_training': cost_adv,
                'adv_for_high_actor_training': adv,
                'High Actor Loss': actor_loss,
                'high_level_unguided_loss': unguided_loss,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample_k_step_trajectory(k=self.k, batch_size=batch_size*4)

        """ POLICY LEARNING """
        actor_high_metrics = self.train_high_actor(state, action)

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()

        return actor_high_metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        state, next_state, action, reward, cost, not_done =\
            replay_buffer.sample_k_step_trajectory(k=self.chunking, batch_size=batch_size)
        reward = self.reward_scale * reward
        bs, seq_len, d = action.shape
        seq_a = action.reshape(bs, -1)

        """ VALUE TRAINING """
        metrics.update(self.train_value(state[:,0], seq_a))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, seq_a, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state[:, 0], seq_a))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, seq_a, cost, not_done))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def sample_sk(self, state):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)
        cond = torch.ones((state.shape[0],), device=state.device)
        with torch.no_grad():
            sk = self.ema_model.sample(state, cond, guidance_scale=3.0) # [100, ACTION_DIM]
        return sk.cpu().data.numpy()

    def sample_action(self, state):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)
        cond = torch.ones((state.shape[0],), device=state.device)
        with torch.no_grad():
            sk = self.ema_model.sample(state, cond, guidance_scale=3.0)
            inputs = torch.cat([state, sk], dim=1)
            action = self.actor_low.sample(inputs, cond, guidance_scale=3.0)
        return action[:,:self.action_dim].cpu().data.numpy()

    def select_action_from_candidates(self, state, eval=True):

        self.ema_model.eval()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=16, dim=0) # [100, STATE_DIM]
        cond_c = torch.ones((state_rpt.shape[0],), device=state.device)
        cond_r = torch.ones((state_rpt.shape[0],), device=state.device)
        if eval:
            with torch.no_grad():
                action = self.ema_model.mix_cond_sample(state_rpt, cond_r, cond_c, guidance_scale=2.0) # [100, ACTION_DIM]

                q_list = self.critic_target(state_rpt, action)
                q_mean = torch.min(torch.stack(q_list), dim=0).values

                v_list = self.value(state_rpt)
                v_mean = torch.mean(torch.stack(v_list), dim=0)
                adv = q_mean - v_mean

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.max(torch.stack(qc_list), dim=0).values

                vc_list = self.cost_value(state_rpt)
                vc_mean = torch.mean(torch.stack(vc_list), dim=0)

                cost_adv = qc_mean - vc_mean #+ self.qc_penalized_coef * torch.std(qc_std**2 + vc_std**2)

                idx = torch.argmin(cost_adv)
        else:
            action = self.ema_model.sample(state_rpt)
            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            action = action[:, :self.action_dim]
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor


    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)


    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list


class FLOWCHUNKWL(object):
    """
    Action chunking + Diffusion + Two-level policy structure
    """
    def __init__(self, state_dim, action_dim, max_action, device, config, saving_logwriter=False):

        self.model = MLPVFieldUnCond(state_dim=state_dim, action_dim=state_dim,
                               device=device, hidden_dim=config['hidden_dim'])

        self.actor = FlowMatchingUnCond(state_dim=state_dim,
                                  action_dim=state_dim,
                                  model=self.model,
                                  max_action=torch.inf,
                                  denoise_steps=config['flow_steps'],).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.model_low = MLPVFieldUnCond(state_dim=state_dim*2,
                                     action_dim=action_dim*config['chunking_length'],
                                     device=device,
                                     hidden_dim=config['hidden_dim'])
        self.actor_low = FlowMatchingUnCond(state_dim=state_dim*2,
                                  action_dim=action_dim*config['chunking_length'],
                                  model=self.model_low,
                                  max_action=max_action,
                                  denoise_steps=config['flow_steps'],).to(device)
        self.actor_low_optimizer = torch.optim.Adam(self.actor_low.parameters(), lr=3e-4)
        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim,
                                          action_dim*config['chunking_length'],
                                          [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim*config['chunking_length'],
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        else:
            raise NotImplementedError

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim,
                                       [512, 512],
                                       nn.Mish, num_v=config['num_v']).to(device)
        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.Mish, num_v=config['num_vc']).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']

        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.k = config['guided_step']
        self.chunking = config['chunking_length']

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_critic(self, state, next_state, action, reward, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(reward[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount**i) * reward[:, i]

            next_value_list = self.value(next_state[:, -1])
            next_value_ = torch.min(torch.stack(next_value_list), dim=0).values
            target = traj_return + not_done[:, -1] * (self.discount**seq_len) * next_value_

        qr_list = self.critic(state[:, 0], action)
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'batch_mean_q': target.mean(),
                'next_value_for_q_training': next_value_,
                'q_loss': critic_loss,}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(cost[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount ** i) * cost[:, i]

            next_value_list = self.cost_value(next_state[:, -1])
            next_value = torch.clamp(torch.max(torch.stack(next_value_list), dim=0).values, min=0)

            target = traj_return + not_done[:, -1] * (self.discount ** seq_len) * next_value

        qc_list = self.cost_critic(state[:,0], action)
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):
        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value_list = self.value(state)
        value_loss_list = []
        for value in value_list:
            u = target_q - value
            expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
            expectile_loss = torch.mean(expectile_weight * u**2)
            value_loss_list.append(expectile_loss)

        value_loss = sum(value_loss_list)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'target_q_for_v_training': target_q,
                'batch_v_in_train_value': torch.max(torch.stack(value_list), dim=0).values,
                'v_loss': value_loss,}

    def train_cost_value(self, state, action):
        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc_list = self.cost_value(state)
        vc_loss_list = []
        for vc in vc_list:
            u = qc - vc
            cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
            expectile_loss = torch.mean(cost_expectile_weight * u ** 2)
            vc_loss_list.append(expectile_loss)

        vc_loss = sum(vc_loss_list)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_high_actor(self, state):
        with torch.no_grad():

            vc_first_list = self.cost_value(state[:,0])
            vc_last_list = self.cost_value(state[:, -1])
            vc_first = torch.max(torch.stack(vc_first_list), dim=0).values
            vc_last = torch.max(torch.stack(vc_last_list), dim=0).values
            cost_adv = (vc_last-vc_first)

            vr_first_list = self.value(state[:, 0])
            vr_last_list = self.value(state[:, -1])
            vr_first = torch.max(torch.stack(vr_first_list), dim=0).values
            vr_last = torch.max(torch.stack(vr_last_list), dim=0).values
            adv = (vr_last - vr_first)

            # condition_0 = torch.zeros_like(adv).squeeze()
            # condition_1 = torch.ones_like(adv).squeeze()
            # mask = (adv > 0).float()
            weight_r = torch.clamp(torch.exp(adv * self.reward_temperature-cost_adv * self.cost_temperature), max=100.0)
            weight_c = torch.clamp(torch.exp(-cost_adv * self.cost_temperature), min=0.0, max=200.0)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        actor_loss = self.actor.loss(state[:, -1], state[:, 0], weights=weight_r)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'cost_adv_for_high_actor_training': cost_adv,
                'adv_for_high_actor_training': adv,
                'High Actor Loss': actor_loss,}
                # 'high_level_guided_loss': guided_loss,
                # 'high_level_unguided_loss': unguided_loss,}

    def train_low_actor(self, state, action):
        bs = state.shape[0]
        seq_a = action[:, :self.chunking].reshape(bs, -1)
        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:,0], seq_a)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_list = self.cost_value(state[:,0])
            vc = torch.max(torch.stack(vc_list), dim=0).values
            cost_adv = (qc - vc)
            exp_b = torch.clamp(torch.exp(-cost_adv*self.cost_temperature), min=0.0, max=200.0)

            # q_list = self.critic_target(state[:,0], seq_a)
            # q = torch.min(torch.stack(q_list), dim=0).values
            # v_list = self.value(state[:,0])
            # v = torch.mean(torch.stack(v_list), dim=0)
            # adv = (q - v)
            # exp_a = torch.exp(adv * self.reward_temperature)
            # exp_a = torch.clamp(exp_a, max=100.0)
            #
            # cond_0 = torch.zeros_like(adv).squeeze()
            # cond_1 = torch.ones_like(adv).squeeze()
            # mask = (cost_adv < 0).float()
            inputs = torch.cat([state[:, 0], state[:, -1]], dim=-1)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        actor_low_loss = self.actor_low.loss(seq_a, inputs, weights=exp_b)

        self.actor_low_optimizer.zero_grad()
        actor_low_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor_low.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_low_optimizer.step()

        return {'cost_adv_for_low_actor_training': cost_adv,
                'Low Actor Loss': actor_low_loss,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample_k_step_trajectory(k=self.k, batch_size=batch_size*4)

        """ POLICY LEARNING """
        actor_high_metrics = self.train_high_actor(state)
        actor_low_metrics = self.train_low_actor(state, action)
        actor_high_metrics.update(actor_low_metrics)

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()

        return actor_high_metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        state, next_state, action, reward, cost, not_done =\
            replay_buffer.sample_k_step_trajectory(k=self.chunking, batch_size=batch_size)
        reward = self.reward_scale * reward
        bs, seq_len, d = action.shape
        seq_a = action.reshape(bs, -1)

        """ VALUE TRAINING """
        metrics.update(self.train_value(state[:,0], seq_a))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, seq_a, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state[:, 0], seq_a))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, seq_a, cost, not_done))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def select_action_from_candidates(self, state, eval=True):

        self.ema_model.eval()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=16, dim=0) # [100, STATE_DIM]
        # cond = torch.ones((state_rpt.shape[0],), device=state.device)
        if eval:
            with torch.no_grad():
                sk = self.ema_model.sample(state_rpt) # [100, ACTION_DIM]
                inputs = torch.cat([state_rpt, sk], dim=1)
                action = self.actor_low.sample(inputs)

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.max(torch.stack(qc_list), dim=0).values

                vc_list = self.cost_value(state_rpt)
                vc_mean = torch.mean(torch.stack(vc_list), dim=0)

                cost_adv = qc_mean - vc_mean #+ self.qc_penalized_coef * torch.std(qc_std**2 + vc_std**2)

                idx = torch.argmin(cost_adv)
        else:
            action = self.ema_model.sample(state_rpt)
            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            action = action[:, :self.action_dim]
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor

    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)

    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list

    def get_vr(self, observations):
        v_list = self.value(observations)
        v = torch.min(torch.stack(v_list), dim=0).values
        return v

    def get_vc(self, observations):
        v_list = self.cost_value(observations)
        v = torch.max(torch.stack(v_list), dim=0).values
        return v


class FLOWCHUNKWLN(object):
    """
    Action chunking + Diffusion + Two-level policy structure + POLICY EXTRACTION MINOR MODIFICATION
    """
    def __init__(self, state_dim, action_dim, max_action, device, config, saving_logwriter=False):

        self.model = MLPVFieldUnCond(state_dim=state_dim, action_dim=state_dim,
                               device=device, hidden_dim=config['hidden_dim'])

        self.actor = FlowMatchingUnCond(state_dim=state_dim,
                                  action_dim=state_dim,
                                  model=self.model,
                                  max_action=torch.inf,
                                  denoise_steps=config['flow_steps'],).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.model_low = MLPVFieldUnCond(state_dim=state_dim*2,
                                     action_dim=action_dim*config['chunking_length'],
                                     device=device,
                                     hidden_dim=config['hidden_dim'])
        self.actor_low = FlowMatchingUnCond(state_dim=state_dim*2,
                                  action_dim=action_dim*config['chunking_length'],
                                  model=self.model_low,
                                  max_action=max_action,
                                  denoise_steps=config['flow_steps'],).to(device)
        self.actor_low_optimizer = torch.optim.Adam(self.actor_low.parameters(), lr=3e-4)

        self.config = config
        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim,
                                          action_dim*config['chunking_length'],
                                          [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim*config['chunking_length'],
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        else:
            raise NotImplementedError

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim,
                                       [512, 512],
                                       nn.Mish, num_v=config['num_v']).to(device)
        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.Mish, num_v=config['num_vc']).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer,
                                                        T_max=config['max_timestep'], eta_min=0.)
            self.actor_low_lr_scheduler = CosineAnnealingLR(self.actor_low_optimizer,
                                                            T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer,
                                                         T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer,
                                                              T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer,
                                                        T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']

        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.k = config['guided_step']
        self.chunking = config['chunking_length']

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_critic(self, state, next_state, action, reward, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(reward[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount**i) * reward[:, i]

            next_value_list = self.value(next_state[:, -1])
            next_value_ = torch.min(torch.stack(next_value_list), dim=0).values
            target = traj_return + not_done[:, -1] * (self.discount**seq_len) * next_value_

        qr_list = self.critic(state[:, 0], action)
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'batch_mean_q': target.mean(),
                'next_value_for_q_training': next_value_,
                'q_loss': critic_loss,}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(cost[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount ** i) * cost[:, i]

            next_value_list = self.cost_value(next_state[:, -1])
            next_value = torch.clamp(torch.max(torch.stack(next_value_list), dim=0).values, min=0)

            target = traj_return + not_done[:, -1] * (self.discount ** seq_len) * next_value

        qc_list = self.cost_critic(state[:,0], action)
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):
        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value_list = self.value(state)
        value_loss_list = []
        for value in value_list:
            u = target_q - value
            expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
            expectile_loss = torch.mean(expectile_weight * u**2)
            value_loss_list.append(expectile_loss)

        value_loss = sum(value_loss_list)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'target_q_for_v_training': target_q,
                'batch_v_in_train_value': torch.max(torch.stack(value_list), dim=0).values,
                'v_loss': value_loss,}

    def train_cost_value(self, state, action):
        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc_list = self.cost_value(state)
        vc_loss_list = []
        for vc in vc_list:
            u = qc - vc
            cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
            expectile_loss = torch.mean(cost_expectile_weight * u ** 2)
            vc_loss_list.append(expectile_loss)

        vc_loss = sum(vc_loss_list)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_high_actor(self, state, action):
        bs = state.shape[0]
        seq_a = action[:, :self.chunking].reshape(bs, -1)

        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:, 0], seq_a)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_first_list = self.cost_value(state[:,0])
            vc_first = torch.max(torch.stack(vc_first_list), dim=0).values
            cost_adv = (qc-vc_first)

            vr_first_list = self.value(state[:, 0])
            vr_last_list = self.value(state[:, -1])
            vr_first = torch.max(torch.stack(vr_first_list), dim=0).values
            vr_last = torch.max(torch.stack(vr_last_list), dim=0).values
            adv = (vr_last - vr_first)

            weight_r = torch.clamp(torch.exp(adv * self.reward_temperature-cost_adv * self.cost_temperature), max=150.0)
            # weight_c = torch.clamp(torch.exp(-cost_adv * self.cost_temperature), min=0.0, max=200.0)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        actor_loss = self.actor.loss(state[:, -1], state[:, 0], weights=weight_r)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'cost_adv_for_high_actor_training': cost_adv,
                'adv_for_high_actor_training': adv,
                'high_actor_cadv_ratio': (cost_adv<0).float().sum()/cost_adv.shape[0],
                'High Actor Loss': actor_loss,}
                # 'high_level_guided_loss': guided_loss,
                # 'high_level_unguided_loss': unguided_loss,}

    def train_low_actor(self, state, action):
        bs = state.shape[0]
        seq_a = action[:, :self.chunking].reshape(bs, -1)
        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:,0], seq_a)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_list = self.cost_value(state[:,0])
            vc = torch.max(torch.stack(vc_list), dim=0).values
            cost_adv = (qc - vc)
            exp_b = torch.clamp(torch.exp(-cost_adv*self.cost_temperature), min=0.0, max=200.0)

            inputs = torch.cat([state[:, 0], state[:, -1]], dim=-1)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        actor_low_loss = self.actor_low.loss(seq_a, inputs, weights=exp_b)

        self.actor_low_optimizer.zero_grad()
        actor_low_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor_low.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_low_optimizer.step()

        return {'cost_adv_for_low_actor_training': cost_adv,
                'Low Actor Loss': actor_low_loss,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample_k_step_trajectory(k=self.k, batch_size=batch_size*4)

        """ POLICY LEARNING """
        actor_high_metrics = self.train_high_actor(state, action)
        actor_low_metrics = self.train_low_actor(state, action)
        actor_high_metrics.update(actor_low_metrics)

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.actor_low_lr_scheduler.step()

        return actor_high_metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        state, next_state, action, reward, cost, not_done =\
            replay_buffer.sample_k_step_trajectory(k=self.chunking, batch_size=batch_size)
        reward = self.reward_scale * reward
        bs, seq_len, d = action.shape
        seq_a = action.reshape(bs, -1)

        """ VALUE TRAINING """
        metrics.update(self.train_value(state[:,0], seq_a))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, seq_a, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state[:, 0], seq_a))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, seq_a, cost, not_done))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def select_action_from_candidates(self, state, eval=True):

        self.ema_model.eval()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=16, dim=0) # [100, STATE_DIM]
        # cond = torch.ones((state_rpt.shape[0],), device=state.device)
        if eval:
            with torch.no_grad():
                sk = self.ema_model.sample(state_rpt) # [100, ACTION_DIM]
                inputs = torch.cat([state_rpt, sk], dim=1)
                action = self.actor_low.sample(inputs)

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.max(torch.stack(qc_list), dim=0).values

                vc_list = self.cost_value(state_rpt)
                vc_mean = torch.mean(torch.stack(vc_list), dim=0)

                cost_adv = qc_mean - vc_mean #+ self.qc_penalized_coef * torch.std(qc_std**2 + vc_std**2)

                idx = torch.argmin(cost_adv)
        else:
            action = self.ema_model.sample(state_rpt)
            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            action = action[:, :self.action_dim]
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor


    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)


    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list

    def get_vr(self, observations):
        v_list = self.value(observations)
        v = torch.min(torch.stack(v_list), dim=0).values
        return v

    def get_vc(self, observations):
        v_list = self.cost_value(observations)
        v = torch.max(torch.stack(v_list), dim=0).values
        return v


class FLOWCHUNKZS(object):
    """
    Action chunking + Diffusion + Two-level policy structure + POLICY EXTRACTION MINOR MODIFICATION
    """
    def __init__(self, state_dim, action_dim, max_action, device, config):

        self.model = MLPVFieldUnCond(state_dim=state_dim, action_dim=state_dim,
                               device=device, hidden_dim=config['hidden_dim'])

        self.actor = FlowMatchingUnCond(state_dim=state_dim,
                                  action_dim=state_dim,
                                  model=self.model,
                                  max_action=torch.inf,
                                  denoise_steps=config['flow_steps'],).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.model_low = MLPVFieldUnCond(state_dim=state_dim*2,
                                     action_dim=action_dim*config['chunking_length'],
                                     device=device,
                                     hidden_dim=config['hidden_dim'])
        self.actor_low = FlowMatchingUnCond(state_dim=state_dim*2,
                                  action_dim=action_dim*config['chunking_length'],
                                  model=self.model_low,
                                  max_action=max_action,
                                  denoise_steps=config['flow_steps'],).to(device)
        self.actor_low_optimizer = torch.optim.Adam(self.actor_low.parameters(), lr=3e-4)

        self.config = config
        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim,
                                          action_dim*config['chunking_length'],
                                          [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim*config['chunking_length'],
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        else:
            raise NotImplementedError

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim,
                                       [512, 512],
                                       nn.GELU, num_v=config['num_v']).to(device)

            self.high_z = EnsembleValue(state_dim,
                                       [512, 512],
                                       nn.GELU, num_v=config['num_v']).to(device)
            self.high_z_optimizer = torch.optim.Adam(self.high_z.parameters(), lr=config['lr'])
            self.low_z = EnsembleValue(state_dim,
                                       [512, 512],
                                       nn.GELU, num_v=config['num_v']).to(device)
            self.low_z_optimizer = torch.optim.Adam(self.low_z.parameters(), lr=config['lr'])

        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.GELU, num_v=config['num_vc']).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer,
                                                        T_max=config['max_timestep'], eta_min=0.)
            self.actor_low_lr_scheduler = CosineAnnealingLR(self.actor_low_optimizer,
                                                            T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer,
                                                         T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer,
                                                              T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer,
                                                        T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']

        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.k = config['guided_step']
        self.chunking = config['chunking_length']

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_critic(self, state, next_state, action, reward, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(reward[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount**i) * reward[:, i]

            next_value_list = self.value(next_state[:, -1])
            next_value_ = torch.min(torch.stack(next_value_list), dim=0).values
            target = traj_return + not_done[:, -1] * (self.discount**seq_len) * next_value_

        qr_list = self.critic(state[:, 0], action)
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'batch_mean_q': target.mean(),
                'next_value_for_q_training': next_value_,
                'q_loss': critic_loss,}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(cost[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount ** i) * cost[:, i]

            next_value_list = self.cost_value(next_state[:, -1])
            next_value = torch.clamp(torch.max(torch.stack(next_value_list), dim=0).values, min=0)

            target = traj_return + not_done[:, -1] * (self.discount ** seq_len) * next_value

        qc_list = self.cost_critic(state[:,0], action)
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):
        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value_list = self.value(state)
        value_loss_list = []
        for value in value_list:
            u = target_q - value
            expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
            expectile_loss = torch.mean(expectile_weight * u**2)
            value_loss_list.append(expectile_loss)

        value_loss = sum(value_loss_list)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'target_q_for_v_training': target_q,
                'batch_v_in_train_value': torch.max(torch.stack(value_list), dim=0).values,
                'v_loss': value_loss,}

    def train_cost_value(self, state, action):
        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc_list = self.cost_value(state)
        vc_loss_list = []
        for vc in vc_list:
            u = qc - vc
            cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
            expectile_loss = torch.mean(cost_expectile_weight * u ** 2)
            vc_loss_list.append(expectile_loss)

        vc_loss = sum(vc_loss_list)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_z(self, state, action):
        with torch.no_grad():
            qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(qc_list), dim=0).values, min=0.)
            vc_list = self.cost_value(state)
            vc = torch.clamp(torch.mean(torch.stack(vc_list), dim=0), min=0.)
            cost_adv = qc - vc

            q_list = self.critic_target(state, action)
            q = torch.min(torch.stack(q_list), dim=0).values
            v_list = self.value(state)
            v = torch.mean(torch.stack(v_list), dim=0)
            adv = q - v

            high_z_target = torch.clamp(torch.exp(adv * self.reward_temperature - cost_adv * self.cost_temperature), max=150.)
            low_z_target = torch.clamp(torch.exp(-cost_adv * self.cost_temperature), max=200.)

        high_z_list = self.high_z(state)
        high_z_loss_list = []
        for high_z in high_z_list:
            high_z_loss = torch.mean((high_z - high_z_target)**2)
            high_z_loss_list.append(high_z_loss)
        high_z_loss = sum(high_z_loss_list)

        low_z_list = self.low_z(state)
        low_z_loss_list = []
        for low_z in low_z_list:
            low_z_loss = torch.mean((low_z - low_z_target)**2)
            low_z_loss_list.append(low_z_loss)
        low_z_loss = sum(low_z_loss_list)

        z_loss = high_z_loss + low_z_loss

        self.high_z_optimizer.zero_grad()
        self.low_z_optimizer.zero_grad()
        z_loss.backward()
        if self.grad_norm > 0:
            high_z_grad_norms = nn.utils.clip_grad_norm_(self.high_z.parameters(), max_norm=self.grad_norm, norm_type=2)
            low_z_grad_norms = nn.utils.clip_grad_norm_(self.low_z.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.high_z_optimizer.step()
        self.low_z_optimizer.step()

        return {'high_z_loss': high_z_loss,
                'low_z_loss': low_z_loss,
                'high_z': high_z_target,
                'low_z': low_z_target}

    def train_high_actor(self, state, action):
        bs = state.shape[0]
        seq_a = action[:, :self.chunking].reshape(bs, -1)

        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:, 0], seq_a)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_first_list = self.cost_value(state[:,0])
            vc_first = torch.max(torch.stack(vc_first_list), dim=0).values
            cost_adv = (qc-vc_first)

            vr_first_list = self.value(state[:, 0])
            vr_last_list = self.value(state[:, -1])
            vr_first = torch.max(torch.stack(vr_first_list), dim=0).values
            vr_last = torch.max(torch.stack(vr_last_list), dim=0).values
            adv = (vr_last - vr_first)

            weight_r = torch.clamp(torch.exp(adv * self.reward_temperature-cost_adv * self.cost_temperature), max=150.0)
            high_z_list = self.high_z(state[:,0])
            high_z = torch.clamp(torch.mean(torch.stack(high_z_list), dim=0), min=1e-3)
            weight_r = weight_r / high_z
            # weight_c = torch.clamp(torch.exp(-cost_adv * self.cost_temperature), min=0.0, max=200.0)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        actor_loss = self.actor.loss(state[:, -1], state[:, 0], weights=weight_r)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'cost_adv_for_high_actor_training': cost_adv,
                'adv_for_high_actor_training': adv,
                'high_actor_cadv_ratio': (cost_adv<0).float().sum()/cost_adv.shape[0],
                'High Actor Loss': actor_loss,}
                # 'high_level_guided_loss': guided_loss,
                # 'high_level_unguided_loss': unguided_loss,}

    def train_low_actor(self, state, action):
        bs = state.shape[0]
        seq_a = action[:, :self.chunking].reshape(bs, -1)
        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:,0], seq_a)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_list = self.cost_value(state[:,0])
            vc = torch.max(torch.stack(vc_list), dim=0).values
            cost_adv = (qc - vc)
            exp_b = torch.clamp(torch.exp(-cost_adv*self.cost_temperature), min=0.0, max=200.0)
            low_z_list = self.low_z(state[:,0])
            low_z = torch.clamp(torch.mean(torch.stack(low_z_list), dim=0), min=1e-3)
            exp_b = exp_b / low_z

            inputs = torch.cat([state[:, 0], state[:, -1]], dim=-1)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        actor_low_loss = self.actor_low.loss(seq_a, inputs, weights=exp_b)

        self.actor_low_optimizer.zero_grad()
        actor_low_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor_low.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_low_optimizer.step()

        return {'cost_adv_for_low_actor_training': cost_adv,
                'Low Actor Loss': actor_low_loss,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample_k_step_trajectory(k=self.k, batch_size=batch_size*4)

        """ POLICY LEARNING """
        actor_high_metrics = self.train_high_actor(state, action)
        actor_low_metrics = self.train_low_actor(state, action)
        actor_high_metrics.update(actor_low_metrics)

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.actor_low_lr_scheduler.step()

        return actor_high_metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        state, next_state, action, reward, cost, not_done =\
            replay_buffer.sample_k_step_trajectory(k=self.chunking, batch_size=batch_size)
        reward = self.reward_scale * reward
        bs, seq_len, d = action.shape
        seq_a = action.reshape(bs, -1)

        """ VALUE TRAINING """
        metrics.update(self.train_value(state[:,0], seq_a))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, seq_a, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state[:, 0], seq_a))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, seq_a, cost, not_done))

        """ Z TRAINING """
        metrics.update(self.train_z(state[:,0], seq_a))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def select_action_from_candidates(self, state, eval=True):

        self.ema_model.eval()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=16, dim=0) # [100, STATE_DIM]
        # cond = torch.ones((state_rpt.shape[0],), device=state.device)
        if eval:
            with torch.no_grad():
                sk = self.ema_model.sample(state_rpt) # [100, ACTION_DIM]
                inputs = torch.cat([state_rpt, sk], dim=1)
                action = self.actor_low.sample(inputs)

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.max(torch.stack(qc_list), dim=0).values

                vc_list = self.cost_value(state_rpt)
                vc_mean = torch.mean(torch.stack(vc_list), dim=0)

                cost_adv = qc_mean - vc_mean #+ self.qc_penalized_coef * torch.std(qc_std**2 + vc_std**2)

                idx = torch.argmin(cost_adv)
        else:
            action = self.ema_model.sample(state_rpt)
            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            action = action[:, :self.action_dim]
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor


    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)


    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list


class FLOWCHUNKNF(object):
    """
    Action chunking + Diffusion + Two-level policy structure + NEARING FUTURE STYLE Q
    """
    def __init__(self, state_dim, action_dim, max_action, device, config):

        self.model = MLPVField(state_dim=state_dim, action_dim=state_dim,
                               device=device, hidden_dim=config['hidden_dim'])

        self.actor = FlowMatching(state_dim=state_dim,
                                  action_dim=state_dim,
                                  model=self.model,
                                  max_action=torch.inf,
                                  denoise_steps=config['flow_steps'],).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.model_low = MLPVFieldUnCond(state_dim=state_dim*2,
                                     action_dim=action_dim*config['chunking_length'],
                                     device=device,
                                     hidden_dim=config['hidden_dim'])
        self.actor_low = FlowMatchingUnCond(state_dim=state_dim*2,
                                  action_dim=action_dim*config['chunking_length'],
                                  model=self.model_low,
                                  max_action=max_action,
                                  denoise_steps=config['flow_steps'],).to(device)
        self.actor_low_optimizer = torch.optim.Adam(self.actor_low.parameters(), lr=3e-4)
        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim,
                                          action_dim*config['chunking_length'],
                                          [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim*config['chunking_length'],
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        else:
            raise NotImplementedError

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim,
                                       [512, 512],
                                       nn.Mish, num_v=config['num_v']).to(device)
        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.Mish, num_v=config['num_vc']).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.config = config
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']

        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.k = config['guided_step']
        self.chunking = config['chunking_length']

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_critic(self, state, next_state, action, reward, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(reward[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount**i) * reward[:, i]

            next_value_list = self.value(next_state[:, -1])
            next_value_ = torch.min(torch.stack(next_value_list), dim=0).values

            vc_list = self.cost_value(next_state[:, -1])
            vc = torch.max(torch.stack(vc_list), dim=0).values
            vc_p10 = torch.quantile(vc, 0.4)
            alpha = 0.01
            self.safety_threshold = (1 - alpha) * self.safety_threshold + alpha * vc_p10
            vc_mask = (vc < self.safety_threshold).float()

            traj_return = vc_mask * traj_return + (1 - vc_mask) * self.config['unsafe_penalty']

            target = traj_return + not_done[:, -1] * vc_mask * (self.discount**seq_len) * next_value_

        qr_list = self.critic(state[:, 0], action)
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'batch_mean_q': target.mean(),
                'next_value_for_q_training': next_value_,
                'q_loss': critic_loss,}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(cost[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount ** i) * cost[:, i]

            next_value_list = self.cost_value(next_state[:, -1])
            next_value = torch.clamp(torch.max(torch.stack(next_value_list), dim=0).values, min=0)

            target = traj_return + not_done[:, -1] * (self.discount ** seq_len) * next_value

        qc_list = self.cost_critic(state[:,0], action)
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):
        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value_list = self.value(state)
        value_loss_list = []
        for value in value_list:
            u = target_q - value
            expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
            expectile_loss = torch.mean(expectile_weight * u**2)
            value_loss_list.append(expectile_loss)

        value_loss = sum(value_loss_list)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'target_q_for_v_training': target_q,
                'batch_v_in_train_value': torch.max(torch.stack(value_list), dim=0).values,
                'v_loss': value_loss,}

    def train_cost_value(self, state, action):
        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc_list = self.cost_value(state)
        vc_loss_list = []
        for vc in vc_list:
            u = qc - vc
            cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
            expectile_loss = torch.mean(cost_expectile_weight * u ** 2)
            vc_loss_list.append(expectile_loss)

        vc_loss = sum(vc_loss_list)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_high_actor(self, state):
        with torch.no_grad():

            # vc_first_list = self.cost_value(state[:,0])
            # vc_last_list = self.cost_value(state[:, -1])
            # vc_first = torch.max(torch.stack(vc_first_list), dim=0).values
            # vc_last = torch.max(torch.stack(vc_last_list), dim=0).values
            # cost_adv = (vc_last-vc_first)

            vr_first_list = self.value(state[:, 0])
            vr_last_list = self.value(state[:, -1])
            vr_first = torch.max(torch.stack(vr_first_list), dim=0).values
            vr_last = torch.max(torch.stack(vr_last_list), dim=0).values
            adv = (vr_last - vr_first)

            condition_0 = torch.zeros_like(adv).squeeze()
            condition_1 = torch.ones_like(adv).squeeze()
            mask = (adv > 0).float()

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        guided_loss = self.actor.loss(state[:, -1], state[:, 0], condition_1, weights=mask)
        unguided_loss = self.actor.loss(state[:, -1], state[:, 0], condition_0)
        actor_loss = guided_loss + 0.1 * unguided_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'adv_for_high_actor_training': adv,
                'High Actor Loss': actor_loss,
                'high_level_guided_loss': guided_loss,
                'high_level_unguided_loss': unguided_loss,}


    def train_low_actor(self, state, action):
        bs = state.shape[0]
        seq_a = action[:, :self.chunking].reshape(bs, -1)
        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:,0], seq_a)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_list = self.cost_value(state[:,0])
            vc = torch.max(torch.stack(vc_list), dim=0).values
            cost_adv = (qc - vc)
            exp_b = torch.clamp(torch.exp(-cost_adv*self.cost_temperature), min=0.0, max=200.0)

            # q_list = self.critic_target(state[:,0], seq_a)
            # q = torch.min(torch.stack(q_list), dim=0).values
            # v_list = self.value(state[:,0])
            # v = torch.mean(torch.stack(v_list), dim=0)
            # adv = (q - v)
            # exp_a = torch.exp(adv * self.reward_temperature)
            # exp_a = torch.clamp(exp_a, max=100.0)

            cond_0 = torch.zeros_like(cost_adv).squeeze()
            cond_1 = torch.ones_like(cost_adv).squeeze()
            mask = (cost_adv < 0).float()
            inputs = torch.cat([state[:, 0], state[:, -1]], dim=-1)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        # guided_loss = self.actor_low.loss(seq_a, inputs, cond_1, weights=mask)
        actor_low_loss = self.actor_low.loss(seq_a, inputs)
        # actor_low_loss = guided_loss + 0.1 * unguided_loss

        self.actor_low_optimizer.zero_grad()
        actor_low_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor_low.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_low_optimizer.step()

        return {'Low Actor Loss': actor_low_loss,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample_k_step_trajectory(k=self.k, batch_size=batch_size*4)

        """ POLICY LEARNING """
        actor_high_metrics = self.train_high_actor(state)
        actor_low_metrics = self.train_low_actor(state, action)
        actor_high_metrics.update(actor_low_metrics)

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()

        return actor_high_metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        state, next_state, action, reward, cost, not_done =\
            replay_buffer.sample_k_step_trajectory(k=self.chunking, batch_size=batch_size)
        reward = self.reward_scale * reward
        bs, seq_len, d = action.shape
        seq_a = action.reshape(bs, -1)

        """ VALUE TRAINING """
        metrics.update(self.train_value(state[:,0], seq_a))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, seq_a, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state[:, 0], seq_a))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, seq_a, cost, not_done))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def select_action_from_candidates(self, state, eval=True):

        self.ema_model.eval()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=16, dim=0) # [100, STATE_DIM]
        cond = torch.ones((state_rpt.shape[0],), device=state.device)
        if eval:
            with torch.no_grad():
                sk = self.ema_model.sample(state_rpt, cond, guidance_scale=2.0) # [100, ACTION_DIM]
                inputs = torch.cat([state_rpt, sk], dim=1)
                action = self.actor_low.sample(inputs)

                # q_list = self.critic_target(state_rpt, action)
                # q_mean = torch.min(torch.stack(q_list), dim=0).values
                #
                # v_list = self.value(state_rpt)
                # v_mean = torch.mean(torch.stack(v_list), dim=0)
                # adv = q_mean - v_mean

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.max(torch.stack(qc_list), dim=0).values

                vc_list = self.cost_value(state_rpt)
                vc_mean = torch.mean(torch.stack(vc_list), dim=0)

                cost_adv = qc_mean - vc_mean #+ self.qc_penalized_coef * torch.std(qc_std**2 + vc_std**2)

                idx = torch.argmin(cost_adv)
        else:
            action = self.ema_model.sample(state_rpt)
            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            action = action[:, :self.action_dim]
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor


    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)


    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list


class FLOWNFS(object):
    """
    Action chunking + Diffusion + NO HIERARCHICAl + ABSORBING REWARD NEO
    """
    def __init__(self, state_dim, action_dim, max_action, device, config, saving_logwriter=False):

        self.model = MLPVField(state_dim=state_dim, action_dim=action_dim*config['chunking_length'],
                               device=device, hidden_dim=config['hidden_dim'])

        self.actor = FlowMatching(state_dim=state_dim,
                                  action_dim=action_dim*config['chunking_length'],
                                  model=self.model,
                                  max_action=max_action,
                                  denoise_steps=config['flow_steps'],).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim,
                                          action_dim*config['chunking_length'],
                                          [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim*config['chunking_length'],
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        else:
            raise NotImplementedError

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim,
                                       [512, 512],
                                       nn.GELU, num_v=config['num_v']).to(device)
        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.GELU,
                                            use_layer_norm=True,
                                            num_v=config['num_vc']).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.config = config
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']

        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.k = config['guided_step']
        self.chunking = config['chunking_length']

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_critic(self, state, next_state, action, reward, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(reward[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount**i) * reward[:, i]

            next_value_list = self.value(next_state[:, -1])
            next_value_ = torch.min(torch.stack(next_value_list), dim=0).values

            vc_list = self.cost_value(next_state[:, -1])
            vc = torch.clamp(torch.max(torch.stack(vc_list), dim=0).values, min=0.)
            vc_mask = (vc<self.safety_threshold).float()

            # vc_p10 = torch.quantile(vc, self.config['safe_portion'])
            # alpha = 0.01  # Learning rate for threshold update
            # self.safety_threshold = (1 - alpha) * self.safety_threshold + alpha * vc_p10.item()
            # vc_mask = (vc < self.safety_threshold).float()

            traj_return = vc_mask * traj_return + (1-vc_mask) * self.config['unsafe_penalty']

            target = traj_return + (not_done[:, -1] * vc_mask * (self.discount**seq_len) * next_value_)

        qr_list = self.critic(state[:, 0], action)
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'batch_mean_q': target.mean(),
                'next_value_for_q_training': next_value_,
                'q_loss': critic_loss,
                'safe_state_ratio': vc_mask.sum()/vc_mask.shape[0],
                'safety_threshold': self.safety_threshold}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(cost[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount ** i) * cost[:, i]

            next_value_list = self.cost_value(next_state[:, -1])
            next_value = torch.clamp(torch.max(torch.stack(next_value_list), dim=0).values, min=0)

            target = traj_return + not_done[:, -1] * (self.discount ** seq_len) * next_value

        qc_list = self.cost_critic(state[:,0], action)
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):
        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value_list = self.value(state)
        value_loss_list = []
        for value in value_list:
            u = target_q - value
            expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
            expectile_loss = torch.mean(expectile_weight * u**2)
            value_loss_list.append(expectile_loss)

        value_loss = sum(value_loss_list)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'target_q_for_v_training': target_q,
                'batch_v_in_train_value': torch.max(torch.stack(value_list), dim=0).values,
                'v_loss': value_loss,}

    def train_cost_value(self, state, action):
        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc_list = self.cost_value(state)
        vc_loss_list = []
        for vc in vc_list:
            u = qc - vc
            cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
            expectile_loss = torch.mean(cost_expectile_weight * u ** 2)
            vc_loss_list.append(expectile_loss)

        vc_loss = sum(vc_loss_list)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_actor(self, state, action):
        bs = state.shape[0]
        seq_a = action[:, :self.chunking].reshape(bs, -1)
        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:,0], seq_a)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_first_list = self.cost_value(state[:,0])
            vc_first = torch.max(torch.stack(vc_first_list), dim=0).values
            cost_adv = (qc-vc_first)

            q_list = self.critic_target(state[:, 0], seq_a)
            q = torch.min(torch.stack(q_list), dim=0).values
            vr_first_list = self.value(state[:, 0])
            vr_first = torch.min(torch.stack(vr_first_list), dim=0).values
            adv = (q - vr_first)

            safe_mask = (vc_first <= self.safety_threshold).float()

            condition_0 = torch.zeros_like(adv).squeeze()
            condition_1 = torch.ones_like(adv).squeeze()
            mask1 = (adv >= 0).float()
            mask2 = (cost_adv <= 0).float()
            weight = mask1 * safe_mask + mask2 * (1 - safe_mask)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        guided_loss = self.actor.loss(seq_a, state[:, 0], condition_1, weights=weight)
        unguided_loss = self.actor.loss(seq_a, state[:, 0], condition_0)
        actor_loss = guided_loss + 0.1 * unguided_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'adv_for_high_actor_training': adv,
                'High Actor Loss': actor_loss,
                'high_level_unguided_loss': unguided_loss,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample_k_step_trajectory(k=self.chunking+1, batch_size=batch_size*4)

        """ POLICY LEARNING """
        actor_high_metrics = self.train_actor(state, action)

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()

        return actor_high_metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        state, next_state, action, reward, cost, not_done =\
            replay_buffer.sample_k_step_trajectory(k=self.chunking, batch_size=batch_size)
        reward = self.reward_scale * reward
        cost = self.cost_scale * cost
        bs, seq_len, d = action.shape
        seq_a = action.reshape(bs, -1)

        """ VALUE TRAINING """
        metrics.update(self.train_value(state[:,0], seq_a))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, seq_a, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state[:, 0], seq_a))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, seq_a, cost, not_done))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def sample_sk(self, state):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)
        cond = torch.ones((state.shape[0],), device=state.device)
        with torch.no_grad():
            sk = self.ema_model.sample(state, cond, guidance_scale=3.0) # [100, ACTION_DIM]
        return sk.cpu().data.numpy()

    def select_action_from_candidates(self, state, eval=True):

        self.ema_model.eval()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=2, dim=0) # [100, STATE_DIM]
        cond_c = torch.ones((state_rpt.shape[0],), device=state.device)
        if eval:
            with torch.no_grad():
                action = self.ema_model.sample(state_rpt, cond_c,
                                               guidance_scale=self.config['guidance_scale']) # [100, ACTION_DIM]

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.max(torch.stack(qc_list), dim=0).values

                q_list = self.critic_target(state_rpt, action)
                q_mean = torch.min(torch.stack(q_list), dim=0).values

                idx = torch.argmin(qc_mean)
        else:
            action = self.ema_model.sample(state_rpt)
            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            action = action[:, :self.action_dim]
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor


    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)


    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list

    def get_vr(self, observations):
        v_list = self.value(observations)
        v = torch.min(torch.stack(v_list), dim=0).values
        return v

    def get_vc(self, observations):
        v_list = self.cost_value(observations)
        v = torch.max(torch.stack(v_list), dim=0).values
        return v


class FLOWNFSW(object):
    """
    Action chunking + Diffusion + NO HIERARCHICAl + ABSORBING REWARD NEO + WEIGHTED LOSS
    """
    def __init__(self, state_dim, action_dim, max_action, device, config, saving_logwriter=False):
        self.actor = Policy(state_dim, action_dim*config['chunking_length'], [512, 512]).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim,
                                          action_dim*config['chunking_length'],
                                          [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim*config['chunking_length'],
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        else:
            raise NotImplementedError

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim,
                                       [512, 512],
                                       nn.GELU, num_v=config['num_v']).to(device)
        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.GELU,
                                            use_layer_norm=True,
                                            num_v=config['num_vc']).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.config = config
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']

        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.k = config['guided_step']
        self.chunking = config['chunking_length']

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_critic(self, state, next_state, action, reward, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(reward[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount**i) * reward[:, i]

            next_value_list = self.value(next_state[:, -1])
            next_value_ = torch.min(torch.stack(next_value_list), dim=0).values

            vc_list = self.cost_value(next_state[:, -1])
            vc = torch.clamp(torch.max(torch.stack(vc_list), dim=0).values, min=0.)
            vc_mask = (vc<self.safety_threshold).float()

            vc_p10 = torch.quantile(vc, self.config['safe_portion'])
            alpha = 0.01  # Learning rate for threshold update
            self.safety_threshold = (1 - alpha) * self.safety_threshold + alpha * vc_p10.item()
            vc_mask = (vc < self.safety_threshold).float()

            traj_return = vc_mask * traj_return + (1-vc_mask) * self.config['unsafe_penalty']

            target = traj_return + (not_done[:, -1] * vc_mask * (self.discount**seq_len) * next_value_)

        qr_list = self.critic(state[:, 0], action)
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'batch_mean_q': target.mean(),
                'next_value_for_q_training': next_value_,
                'q_loss': critic_loss,
                'safe_state_ratio': vc_mask.sum()/vc_mask.shape[0],
                'safety_threshold': self.safety_threshold}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(cost[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount ** i) * cost[:, i]

            next_value_list = self.cost_value(next_state[:, -1])
            next_value = torch.clamp(torch.max(torch.stack(next_value_list), dim=0).values, min=0)

            target = traj_return + not_done[:, -1] * (self.discount ** seq_len) * next_value

        qc_list = self.cost_critic(state[:,0], action)
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):
        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value_list = self.value(state)
        value_loss_list = []
        for value in value_list:
            u = target_q - value
            expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
            expectile_loss = torch.mean(expectile_weight * u**2)
            value_loss_list.append(expectile_loss)

        value_loss = sum(value_loss_list)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'target_q_for_v_training': target_q,
                'batch_v_in_train_value': torch.max(torch.stack(value_list), dim=0).values,
                'v_loss': value_loss,}

    def train_cost_value(self, state, action):
        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc_list = self.cost_value(state)
        vc_loss_list = []
        for vc in vc_list:
            u = qc - vc
            cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
            expectile_loss = torch.mean(cost_expectile_weight * u ** 2)
            vc_loss_list.append(expectile_loss)

        vc_loss = sum(vc_loss_list)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_actor(self, state, action):
        bs = state.shape[0]
        seq_a = action[:, :self.chunking].reshape(bs, -1)
        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:,0], seq_a)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_first_list = self.cost_value(state[:,0])
            vc_first = torch.max(torch.stack(vc_first_list), dim=0).values
            cost_adv = (qc-vc_first)

            q_list = self.critic_target(state[:, 0], seq_a)
            q = torch.min(torch.stack(q_list), dim=0).values
            vr_first_list = self.value(state[:, 0])
            vr_first = torch.min(torch.stack(vr_first_list), dim=0).values
            adv = (q - vr_first)

            safe_mask = (vc_first <= self.safety_threshold).float()

            weight_r = torch.clamp(torch.exp(self.reward_temperature * adv), max=150.)
            weight_c = torch.clamp(torch.exp(-self.cost_temperature * cost_adv), max=200.)
            weight = weight_r * safe_mask + weight_c * (1 - safe_mask)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        dist = self.actor(state[:,0])
        log_probs = dist.log_prob(seq_a).sum(-1)
        actor_loss = - (weight * log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'adv_for_high_actor_training': adv,
                'High Actor Loss': actor_loss,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample_k_step_trajectory(k=self.chunking+1, batch_size=batch_size*4)

        """ POLICY LEARNING """
        actor_high_metrics = self.train_actor(state, action)

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()

        return actor_high_metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        state, next_state, action, reward, cost, not_done =\
            replay_buffer.sample_k_step_trajectory(k=self.chunking, batch_size=batch_size)
        reward = self.reward_scale * reward
        cost = self.cost_scale * cost
        bs, seq_len, d = action.shape
        seq_a = action.reshape(bs, -1)

        """ VALUE TRAINING """
        metrics.update(self.train_value(state[:,0], seq_a))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, seq_a, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state[:, 0], seq_a))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, seq_a, cost, not_done))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def sample_sk(self, state):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)
        cond = torch.ones((state.shape[0],), device=state.device)
        with torch.no_grad():
            sk = self.ema_model.sample(state, cond, guidance_scale=3.0) # [100, ACTION_DIM]
        return sk.cpu().data.numpy()

    def select_action_from_candidates(self, state, eval=True):

        self.ema_model.eval()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=16, dim=0) # [100, STATE_DIM]
        cond_c = torch.ones((state_rpt.shape[0],), device=state.device)
        if eval:
            with torch.no_grad():
                dist = self.actor(state_rpt)
                action = dist.rsample()

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.max(torch.stack(qc_list), dim=0).values

                vc_list = self.cost_value(state_rpt)
                vc_mean = torch.mean(torch.stack(vc_list), dim=0)

                cost_adv = qc_mean - vc_mean #+ self.qc_penalized_coef * torch.std(qc_std**2 + vc_std**2)

                idx = torch.argmin(cost_adv)
        else:
            action = self.ema_model.sample(state_rpt)
            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            action = action[:, :self.action_dim]
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor


    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)


    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list

    def get_vr(self, observations):
        v_list = self.value(observations)
        v = torch.min(torch.stack(v_list), dim=0).values
        return v

    def get_vc(self, observations):
        v_list = self.cost_value(observations)
        v = torch.max(torch.stack(v_list), dim=0).values
        return v


class FLOWNFWF(object):
    """
    Action chunking + Diffusion + NO HIERARCHICAl + ABSORBING REWARD NEO
    """
    def __init__(self, state_dim, action_dim, max_action, device, config, saving_logwriter=False):

        self.model = MLPVFieldUnCond(state_dim=state_dim, action_dim=action_dim*config['chunking_length'],
                               device=device, hidden_dim=config['hidden_dim'])

        self.actor = FlowMatchingUnCond(state_dim=state_dim,
                                  action_dim=action_dim*config['chunking_length'],
                                  model=self.model,
                                  max_action=max_action,
                                  denoise_steps=config['flow_steps'],).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim,
                                          action_dim*config['chunking_length'],
                                          [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim*config['chunking_length'],
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        else:
            raise NotImplementedError

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim,
                                       [512, 512],
                                       nn.GELU, num_v=config['num_v']).to(device)
        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.GELU,
                                            use_layer_norm=True,
                                            num_v=config['num_vc']).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.config = config
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']

        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.k = config['guided_step']
        self.chunking = config['chunking_length']

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_critic(self, state, next_state, action, reward, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(reward[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount**i) * reward[:, i]

            next_value_list = self.value(next_state[:, -1])
            next_value_ = torch.min(torch.stack(next_value_list), dim=0).values

            vc_list = self.cost_value(next_state[:, -1])
            vc = torch.clamp(torch.max(torch.stack(vc_list), dim=0).values, min=0.)
            vc_mask = (vc<self.safety_threshold).float()

            vc_p10 = torch.quantile(vc, self.config['safe_portion'])
            alpha = 0.01  # Learning rate for threshold update
            self.safety_threshold = (1 - alpha) * self.safety_threshold + alpha * vc_p10.item()
            vc_mask = (vc < self.safety_threshold).float()

            traj_return = vc_mask * traj_return + (1-vc_mask) * self.config['unsafe_penalty']

            target = traj_return + (not_done[:, -1] * vc_mask * (self.discount**seq_len) * next_value_)

        qr_list = self.critic(state[:, 0], action)
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'batch_mean_q': target.mean(),
                'next_value_for_q_training': next_value_,
                'q_loss': critic_loss,
                'safe_state_ratio': vc_mask.sum()/vc_mask.shape[0],
                'safety_threshold': self.safety_threshold}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(cost[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount ** i) * cost[:, i]

            next_value_list = self.cost_value(next_state[:, -1])
            next_value = torch.clamp(torch.max(torch.stack(next_value_list), dim=0).values, min=0)

            target = traj_return + not_done[:, -1] * (self.discount ** seq_len) * next_value

        qc_list = self.cost_critic(state[:,0], action)
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):
        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value_list = self.value(state)
        value_loss_list = []
        for value in value_list:
            u = target_q - value
            expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
            expectile_loss = torch.mean(expectile_weight * u**2)
            value_loss_list.append(expectile_loss)

        value_loss = sum(value_loss_list)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'target_q_for_v_training': target_q,
                'batch_v_in_train_value': torch.max(torch.stack(value_list), dim=0).values,
                'v_loss': value_loss,}

    def train_cost_value(self, state, action):
        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc_list = self.cost_value(state)
        vc_loss_list = []
        for vc in vc_list:
            u = qc - vc
            cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
            expectile_loss = torch.mean(cost_expectile_weight * u ** 2)
            vc_loss_list.append(expectile_loss)

        vc_loss = sum(vc_loss_list)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_actor(self, state, action):
        bs = state.shape[0]
        seq_a = action[:, :self.chunking].reshape(bs, -1)
        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:,0], seq_a)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_first_list = self.cost_value(state[:,0])
            vc_first = torch.max(torch.stack(vc_first_list), dim=0).values
            cost_adv = (qc-vc_first)

            q_list = self.critic_target(state[:, 0], seq_a)
            q = torch.min(torch.stack(q_list), dim=0).values
            vr_first_list = self.value(state[:, 0])
            vr_first = torch.min(torch.stack(vr_first_list), dim=0).values
            adv = (q - vr_first)

            safe_mask = (vc_first <= self.safety_threshold).float()

            mask1 = torch.clamp(torch.exp(self.reward_temperature * adv), max=150.)
            mask2 = torch.clamp(torch.exp(-self.cost_temperature * cost_adv), max=200.)
            weight = mask1 * safe_mask + mask2 * (1 - safe_mask)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        guided_loss = self.actor.loss(seq_a, state[:, 0], weights=weight)
        actor_loss = guided_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'adv_for_high_actor_training': adv,
                'High Actor Loss': actor_loss,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample_k_step_trajectory(k=self.chunking+1, batch_size=batch_size*4)

        """ POLICY LEARNING """
        actor_high_metrics = self.train_actor(state, action)

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()

        return actor_high_metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        state, next_state, action, reward, cost, not_done =\
            replay_buffer.sample_k_step_trajectory(k=self.chunking, batch_size=batch_size)
        reward = self.reward_scale * reward
        cost = self.cost_scale * cost
        bs, seq_len, d = action.shape
        seq_a = action.reshape(bs, -1)

        """ VALUE TRAINING """
        metrics.update(self.train_value(state[:,0], seq_a))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, seq_a, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state[:, 0], seq_a))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, seq_a, cost, not_done))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def sample_sk(self, state):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)
        cond = torch.ones((state.shape[0],), device=state.device)
        with torch.no_grad():
            sk = self.ema_model.sample(state, cond, guidance_scale=3.0) # [100, ACTION_DIM]
        return sk.cpu().data.numpy()

    def select_action_from_candidates(self, state, eval=True):

        self.ema_model.eval()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=16, dim=0) # [100, STATE_DIM]
        cond_c = torch.ones((state_rpt.shape[0],), device=state.device)
        if eval:
            with torch.no_grad():
                action = self.ema_model.sample(state_rpt) # [100, ACTION_DIM]

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.max(torch.stack(qc_list), dim=0).values

                vc_list = self.cost_value(state_rpt)
                vc_mean = torch.mean(torch.stack(vc_list), dim=0)

                cost_adv = qc_mean - vc_mean #+ self.qc_penalized_coef * torch.std(qc_std**2 + vc_std**2)

                idx = torch.argmin(cost_adv)
        else:
            action = self.ema_model.sample(state_rpt)
            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            action = action[:, :self.action_dim]
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor


    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)


    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list

    def get_vr(self, observations):
        v_list = self.value(observations)
        v = torch.min(torch.stack(v_list), dim=0).values
        return v

    def get_vc(self, observations):
        v_list = self.cost_value(observations)
        v = torch.max(torch.stack(v_list), dim=0).values
        return v


class FLOWCHUNKNFS(object):
    """
    Action chunking + Diffusion + Two-level policy structure + NEARING FUTURE STYLE Q + SAFE ACTOR
    """
    def __init__(self, state_dim, action_dim, max_action, device, config):

        self.model = MLPVField(state_dim=state_dim, action_dim=state_dim,
                               device=device, hidden_dim=config['hidden_dim'])

        self.actor = FlowMatching(state_dim=state_dim,
                                  action_dim=state_dim,
                                  model=self.model,
                                  max_action=torch.inf,
                                  denoise_steps=config['flow_steps'],).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.model_low = MLPVFieldUnCond(state_dim=state_dim*2,
                                     action_dim=action_dim*config['chunking_length'],
                                     device=device,
                                     hidden_dim=config['hidden_dim'])
        self.actor_low = FlowMatchingUnCond(state_dim=state_dim*2,
                                  action_dim=action_dim*config['chunking_length'],
                                  model=self.model_low,
                                  max_action=max_action,
                                  denoise_steps=config['flow_steps'],).to(device)
        self.actor_low_optimizer = torch.optim.Adam(self.actor_low.parameters(), lr=3e-4)
        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim,
                                          action_dim*config['chunking_length'],
                                          [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim*config['chunking_length'],
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        else:
            raise NotImplementedError

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim,
                                       [512, 512],
                                       nn.Mish, num_v=config['num_v']).to(device)
        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.Mish, num_v=config['num_vc']).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.config = config
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']

        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.k = config['guided_step']
        self.chunking = config['chunking_length']

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)


    def train_critic(self, state, next_state, action, reward, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(reward[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount**i) * reward[:, i]

            next_value_list = self.value(next_state[:, -1])
            next_value_ = torch.min(torch.stack(next_value_list), dim=0).values

            vc_list = self.cost_value(next_state[:, -1])
            vc = torch.clamp(torch.max(torch.stack(vc_list), dim=0).values, min=0.)
            vc_mask = (vc<self.safety_threshold).float()

            vc_p10 = torch.quantile(vc, self.config['safe_portion'])
            alpha = 0.01  # Learning rate for threshold update
            self.safety_threshold = (1 - alpha) * self.safety_threshold + alpha * vc_p10.item()
            vc_mask = (vc < self.safety_threshold).float()

            target = (traj_return + not_done[:, -1] * (self.discount**seq_len) * next_value_) * vc_mask + \
                     self.config['unsafe_penalty'] * (1 - vc_mask)

        qr_list = self.critic(state[:, 0], action)
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'batch_mean_q': target.mean(),
                'next_value_for_q_training': next_value_,
                'q_loss': critic_loss,
                'safe_state_ratio': vc_mask.sum()/vc_mask.shape[0],
                'safety_threshold': self.safety_threshold}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(cost[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount ** i) * cost[:, i]

            next_value_list = self.cost_value(next_state[:, -1])
            next_value = torch.clamp(torch.max(torch.stack(next_value_list), dim=0).values, min=0)

            target = traj_return + not_done[:, -1] * (self.discount ** seq_len) * next_value

        qc_list = self.cost_critic(state[:,0], action)
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):
        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value_list = self.value(state)
        value_loss_list = []
        for value in value_list:
            u = target_q - value
            expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
            expectile_loss = torch.mean(expectile_weight * u**2)
            value_loss_list.append(expectile_loss)

        value_loss = sum(value_loss_list)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'target_q_for_v_training': target_q,
                'batch_v_in_train_value': torch.max(torch.stack(value_list), dim=0).values,
                'v_loss': value_loss,}

    def train_cost_value(self, state, action):
        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc_list = self.cost_value(state)
        vc_loss_list = []
        for vc in vc_list:
            u = qc - vc
            cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
            expectile_loss = torch.mean(cost_expectile_weight * u ** 2)
            vc_loss_list.append(expectile_loss)

        vc_loss = sum(vc_loss_list)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_high_actor(self, state):
        with torch.no_grad():

            vc_first_list = self.cost_value(state[:,0])
            vc_last_list = self.cost_value(state[:, -1])
            vc_first = torch.max(torch.stack(vc_first_list), dim=0).values
            vc_last = torch.max(torch.stack(vc_last_list), dim=0).values
            cost_adv = (vc_last-vc_first)

            vr_first_list = self.value(state[:, 0])
            vr_last_list = self.value(state[:, -1])
            vr_first = torch.max(torch.stack(vr_first_list), dim=0).values
            vr_last = torch.max(torch.stack(vr_last_list), dim=0).values
            adv = (vr_last - vr_first)

            safe_mask = (vc_first < self.safety_threshold).float()

            condition_0 = torch.zeros_like(adv).squeeze()
            condition_1 = torch.ones_like(adv).squeeze()
            mask1 = (adv >= 0).float()
            mask2 = (cost_adv <= 0).float()
            weight_loss = safe_mask * mask1 + (1 - safe_mask) * mask2

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        guided_loss = self.actor.loss(state[:, -1], state[:, 0], condition_1, weights=weight_loss)
        unguided_loss = self.actor.loss(state[:, -1], state[:, 0], condition_0)
        actor_loss = guided_loss + 0.1 * unguided_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'adv_for_high_actor_training': adv,
                'High Actor Loss': actor_loss,
                'high_level_guided_loss': guided_loss,
                'high_level_unguided_loss': unguided_loss,}


    def train_low_actor(self, state, action):
        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:,0], action)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc_list = self.cost_value(state[:,0])
            vc = torch.max(torch.stack(vc_list), dim=0).values
            cost_adv = (qc - vc)
            exp_b = torch.clamp(torch.exp(-cost_adv*self.cost_temperature), min=0.0, max=200.0)

            # q_list = self.critic_target(state[:,0], seq_a)
            # q = torch.min(torch.stack(q_list), dim=0).values
            # v_list = self.value(state[:,0])
            # v = torch.mean(torch.stack(v_list), dim=0)
            # adv = (q - v)
            # exp_a = torch.exp(adv * self.reward_temperature)
            # exp_a = torch.clamp(exp_a, max=100.0)

            cond_0 = torch.zeros_like(cost_adv).squeeze()
            cond_1 = torch.ones_like(cost_adv).squeeze()
            mask = (cost_adv < 0).float()
            inputs = torch.cat([state[:, 0], state[:, -1]], dim=-1)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        # guided_loss = self.actor_low.loss(seq_a, inputs, cond_1, weights=mask)
        actor_low_loss = self.actor_low.loss(action, inputs)
        # actor_low_loss = guided_loss + 0.1 * unguided_loss

        self.actor_low_optimizer.zero_grad()
        actor_low_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor_low.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_low_optimizer.step()

        return {'Low Actor Loss': actor_low_loss,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample_k_step_trajectory(k=self.k, batch_size=batch_size*4)

        """ POLICY LEARNING """
        bs = state.shape[0]
        seq_a = action[:,:self.chunking].reshape(bs, -1)
        actor_high_metrics = self.train_high_actor(state)
        actor_low_metrics = self.train_low_actor(state, seq_a)
        actor_high_metrics.update(actor_low_metrics)

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()

        return actor_high_metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        state, next_state, action, reward, cost, not_done =\
            replay_buffer.sample_k_step_trajectory(k=self.chunking, batch_size=batch_size)
        reward = self.reward_scale * reward
        bs, seq_len, d = action.shape
        seq_a = action.reshape(bs, -1)

        """ VALUE TRAINING """
        metrics.update(self.train_value(state[:,0], seq_a))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, seq_a, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state[:, 0], seq_a))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, seq_a, cost, not_done))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def select_action_from_candidates(self, state, eval=True):

        self.ema_model.eval()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=16, dim=0) # [100, STATE_DIM]
        cond = torch.ones((state_rpt.shape[0],), device=state.device)
        if eval:
            with torch.no_grad():
                sk = self.ema_model.sample(state_rpt, cond, guidance_scale=2.0) # [100, ACTION_DIM]
                inputs = torch.cat([state_rpt, sk], dim=1)
                action = self.actor_low.sample(inputs)

                # q_list = self.critic_target(state_rpt, action)
                # q_mean = torch.min(torch.stack(q_list), dim=0).values
                #
                # v_list = self.value(state_rpt)
                # v_mean = torch.mean(torch.stack(v_list), dim=0)
                # adv = q_mean - v_mean

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.max(torch.stack(qc_list), dim=0).values

                vc_list = self.cost_value(state_rpt)
                vc_mean = torch.mean(torch.stack(vc_list), dim=0)

                cost_adv = qc_mean - vc_mean #+ self.qc_penalized_coef * torch.std(qc_std**2 + vc_std**2)

                idx = torch.argmin(cost_adv)
        else:
            action = self.ema_model.sample(state_rpt)
            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            action = action[:, :self.action_dim]
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor


    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'actor_low_state_dict': self.actor_low.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)


    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.actor_low.load_state_dict(checkpoint['actor_low_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list


class FLOWNFSFEASI(object):
    """
    Action chunking + Diffusion + NO HIERARCHICAl + ABSORBING REWARD NEO + FEASIBILITY
    """
    def __init__(self, state_dim, action_dim, max_action, device, config, saving_logwriter=False):

        self.model = MLPVField(state_dim=state_dim, action_dim=action_dim*config['chunking_length'],
                               device=device, hidden_dim=config['hidden_dim'])

        self.actor = FlowMatching(state_dim=state_dim,
                                  action_dim=action_dim*config['chunking_length'],
                                  model=self.model,
                                  max_action=max_action,
                                  denoise_steps=config['flow_steps'],).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.lr_decay = config['lr_decay']
        self.grad_norm = config['gn']

        self.step = 0
        self.ema = EMA(config['ema_decay'])
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = config['update_ema_every']

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.critic = EnsembleQCritic(state_dim,
                                          action_dim*config['chunking_length'],
                                          [512, 512],
                                          nn.GELU, num_q=config['num_q']).to(device)
            self.critic_target = copy.deepcopy(self.critic)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['lr'])

            self.cost_critic = EnsembleQCritic(state_dim,
                                               action_dim*config['chunking_length'],
                                               [512, 512],
                                               nn.GELU,
                                               num_q=config['num_qc']).to(device)
            self.cost_critic_target = copy.deepcopy(self.cost_critic)
            self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=config['lr'])
        else:
            raise NotImplementedError

        ################################################################################################################
        if config['critic_net'] == 'mlp':
            self.value = EnsembleValue(state_dim,
                                       [512, 512],
                                       nn.GELU, num_v=1).to(device)
        else:
            raise NotImplementedError
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config['lr'])

        if config['critic_net'] == 'mlp':
            self.cost_value = EnsembleValue(state_dim, [512, 512], nn.GELU,
                                            use_layer_norm=True,
                                            num_v=1).to(device)
        else:
            raise NotImplementedError
        self.cost_value_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=config['lr'])
        ################################################################################################################

        if self.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_critic_lr_scheduler = CosineAnnealingLR(self.cost_critic_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.value_lr_scheduler = CosineAnnealingLR(self.value_optimizer, T_max=config['max_timestep'], eta_min=0.)
            self.cost_value_lr_scheduler = CosineAnnealingLR(self.cost_value_optimizer, T_max=config['max_timestep'], eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.config = config
        self.discount = config['gamma']
        self.expectile_temp = config['expectile_temp']
        self.cost_expectile_temp = config['cost_expectile_temp']
        self.max_timestep = config['max_timestep']

        self.tau = config['tau']
        self.eta = config['eta']  # q_learning weight
        self.device = device
        self.max_q_backup = config['max_q_backup']

        self.cost_temperature = config['cost_temperature']
        self.reward_temperature = config['reward_temperature']
        self.cost_adv_ub = config['cost_adv_ub']
        self.cost_scale = config['cost_scale']
        self.reward_scale = config['reward_scale']

        self.print_more = config['print_more_info']

        self.noise_scale = config['ood_noise']

        self.episode_len = config['episode_length']
        self.safety_threshold = 10. * (1 - self.discount**self.episode_len) / (
            1 - self.discount) / self.episode_len

        self.k = config['guided_step']
        self.chunking = config['chunking_length']

    def step_ema(self):
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_critic(self, state, next_state, action, reward, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(reward[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount**i) * reward[:, i]

            next_value_ = self.value(next_state[:, -1])[0]

            vc = self.cost_value(next_state[:, -1])[0]
            vc_mask = (vc<0).float()

            traj_return = vc_mask * traj_return + (1-vc_mask) * self.config['unsafe_penalty']

            target = traj_return + (not_done[:, -1] * vc_mask * (self.discount**seq_len) * next_value_)

        qr_list = self.critic(state[:, 0], action)
        # bellman_loss = self.critic.loss(target, qr_list)
        critic_loss_list = []
        for qr in qr_list:
            bellman_loss = torch.mean((qr - target)**2)
            critic_loss_list.append(bellman_loss)

        critic_loss = sum(critic_loss_list)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.grad_norm > 0:
            critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.critic_optimizer.step()

        # UPDATE TARGET CRITIC
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_for_q_training': target,
                'batch_mean_q': target.mean(),
                'next_value_for_q_training': next_value_,
                'q_loss': critic_loss,
                'safe_state_ratio': vc_mask.sum()/vc_mask.shape[0],
                'safety_threshold': self.safety_threshold}

    def train_cost_critic(self, state, next_state, action, cost, not_done):
        bs, seq_len, d = state.shape
        with torch.no_grad():
            traj_return = torch.zeros_like(cost[:, 0])
            for i in range(seq_len):
                traj_return += (self.discount ** i) * cost[:, i]

            next_value = self.cost_value(next_state[:, -1])[0]
            traj_return = torch.where(traj_return > 0.1, torch.tensor(25).to('cuda'), torch.tensor(-1).to('cuda'))

            target = (1-self.discount) * traj_return + not_done[:, -1] * \
                     (self.discount ** seq_len) * torch.max(traj_return, next_value)

        qc_list = self.cost_critic(state[:,0], action)
        cost_critic_loss_list = []
        for qc in qc_list:
            bellman_loss = torch.mean((qc - target)**2)
            cost_critic_loss_list.append(bellman_loss)

        cost_critic_loss = sum(cost_critic_loss_list)

        self.cost_critic_optimizer.zero_grad()
        cost_critic_loss.backward()
        if self.grad_norm > 0:
            cost_critic_grad_norms = nn.utils.clip_grad_norm_(self.cost_critic.parameters(),
                                                              max_norm=self.grad_norm, norm_type=2)
        self.cost_critic_optimizer.step()

        # UPDATE TARGET COST CRITIC
        for param, target_param in zip(self.cost_critic.parameters(), self.cost_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {'target_qc_for_qc_training': target,
                'cost_value_for_qc_training': next_value,
                'batch_mean_qc': target.mean(),
                'qc_loss': cost_critic_loss,}

    def train_value(self, state, action):
        with torch.no_grad():
            target_q_list = self.critic_target(state, action)
            target_q = torch.min(torch.stack(target_q_list), dim=0).values

        value = self.value(state)[0]
        u = target_q - value
        expectile_weight = torch.abs(self.expectile_temp - (u<0).float())
        value_loss = torch.mean(expectile_weight * u**2)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.grad_norm > 0:
            value_grad_norms = nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=self.grad_norm,
                                                        norm_type=2)
        self.value_optimizer.step()

        return {'target_q_for_v_training': target_q,
                'v_loss': value_loss,}

    def train_cost_value(self, state, action):
        with torch.no_grad():
            target_qc_list = self.cost_critic_target(state, action)
            qc = torch.clamp(torch.max(torch.stack(target_qc_list), dim=0).values, min=0.)

        vc = self.cost_value(state)[0]
        u = qc - vc
        cost_expectile_weight = torch.abs(self.cost_expectile_temp - (u > 0).float())
        vc_loss = torch.mean(cost_expectile_weight * u ** 2)

        self.cost_value_optimizer.zero_grad()
        vc_loss.backward()
        if self.grad_norm > 0:
            cost_value_grad_norms = nn.utils.clip_grad_norm_(self.cost_value.parameters(),
                                                             max_norm=self.grad_norm, norm_type=2)
        self.cost_value_optimizer.step()

        return {'target_qc_for_vc_training': qc,
                'vc_loss': vc_loss,}

    def train_actor(self, state, action):
        bs = state.shape[0]
        seq_a = action[:, :self.chunking].reshape(bs, -1)
        with torch.no_grad():
            qc_list = self.cost_critic_target(state[:,0], seq_a)
            qc = torch.mean(torch.stack(qc_list), dim=0)
            vc = self.cost_value(state[:,0])[0]
            cost_adv = (qc-vc)

            q_list = self.critic_target(state[:, 0], seq_a)
            q = torch.min(torch.stack(q_list), dim=0).values
            vr = self.value(state[:, 0])[0]
            adv = (q - vr)

            safe_mask = (vc <= 0).float()

            condition_0 = torch.zeros_like(adv).squeeze()
            condition_1 = torch.ones_like(adv).squeeze()
            mask1 = (adv >= 0).float()
            mask2 = (cost_adv <= 0).float()
            weight = mask1 * safe_mask + mask2 * (1 - safe_mask)

        # USE WEIGHTS TO WEIGHT LOSS IN DIFFUSION STEP
        guided_loss = self.actor.loss(seq_a, state[:, 0], condition_1, weights=weight)
        unguided_loss = self.actor.loss(seq_a, state[:, 0], condition_0)
        actor_loss = guided_loss + 0.1 * unguided_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(),
                                                        max_norm=self.grad_norm,
                                                        norm_type=2)
        self.actor_optimizer.step()

        return {'adv_for_high_actor_training': adv,
                'High Actor Loss': actor_loss,
                'high_level_unguided_loss': unguided_loss,}

    def train_actor_only(self, replay_buffer, gradient_step, batch_size=100,):
        # UES NORMAL BATCH SIZE (256) FOR CRITIC TRAINING
        state, next_state, action, reward, cost, not_done = \
            replay_buffer.sample_k_step_trajectory(k=self.chunking+1, batch_size=batch_size*4)

        """ POLICY LEARNING """
        actor_high_metrics = self.train_actor(state, action)

        if gradient_step % self.update_ema_every == 0:
            # UPDATE ACTOR
            self.step_ema()

        if self.lr_decay:
            self.actor_lr_scheduler.step()

        return actor_high_metrics

    def train_critic_only(self, replay_buffer, batch_size=100):
        metrics = {}
        state, next_state, action, reward, cost, not_done =\
            replay_buffer.sample_k_step_trajectory(k=self.chunking, batch_size=batch_size)
        reward = self.reward_scale * reward
        cost = self.cost_scale * cost
        bs, seq_len, d = action.shape
        seq_a = action.reshape(bs, -1)

        """ VALUE TRAINING """
        metrics.update(self.train_value(state[:,0], seq_a))

        """ Q TRAINING """
        metrics.update(self.train_critic(state, next_state, seq_a, reward, not_done))

        """ COST VALUE TRAINING """
        metrics.update(self.train_cost_value(state[:, 0], seq_a))

        """ COST Q TRAINING """
        metrics.update(self.train_cost_critic(state, next_state, seq_a, cost, not_done))

        if self.lr_decay:
            self.critic_lr_scheduler.step()
            self.value_lr_scheduler.step()
            self.cost_critic_lr_scheduler.step()
            self.cost_value_lr_scheduler.step()

        """ LOG INFO """
        return metrics

    def sample_sk(self, state):
        if not torch.is_tensor(state):
            state = torch.FloatTensor(state).to(self.device)
        cond = torch.ones((state.shape[0],), device=state.device)
        with torch.no_grad():
            sk = self.ema_model.sample(state, cond, guidance_scale=3.0) # [100, ACTION_DIM]
        return sk.cpu().data.numpy()

    def select_action_from_candidates(self, state, eval=True):

        self.ema_model.eval()

        if not torch.is_tensor(state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        input_dim = state.shape[0]
        state_rpt = torch.repeat_interleave(state, repeats=16, dim=0) # [100, STATE_DIM]
        cond_c = torch.ones((state_rpt.shape[0],), device=state.device)
        if eval:
            with torch.no_grad():
                action = self.ema_model.sample(state_rpt, cond_c,
                                               guidance_scale=self.config['guidance_scale']) # [100, ACTION_DIM]

                qc_list = self.cost_critic_target(state_rpt, action)
                qc_mean = torch.max(torch.stack(qc_list), dim=0).values

                vc_mean = self.cost_value(state_rpt)[0]

                cost_adv = qc_mean - vc_mean #+ self.qc_penalized_coef * torch.std(qc_std**2 + vc_std**2)

                idx = torch.argmin(cost_adv)
        else:
            action = self.ema_model.sample(state_rpt)
            q_value = self.critic_target.predict(state_rpt, action)[0].flatten().reshape(input_dim, -1)
            idx = torch.multinomial(F.softmax(q_value), 1)
        if input_dim == 1:
            action = action[:, :self.action_dim]
            re_action = action[idx].clip(-1, 1)

            self.ema_model.train()

            return re_action.cpu().data.numpy().flatten()  # Single input return numpy
        else:
            re_action = torch.index_select(action.reshape(input_dim, 50, -1), 1, idx.reshape(-1))
            re_action = torch.diagonal(re_action, dim1=0, dim2=1).T
            re_q = torch.index_select(q_value, 1, idx.reshape(-1))
            re_q = torch.diagonal(re_q)

            self.ema_model.train()

            return re_action.reshape(input_dim, -1) # Multi input return torch.tensor


    def save_model(self, file_name):
        logger.info('Saving models to {}'.format(file_name))
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'ema_state_dict': self.ema_model.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'value_state_dict': self.value.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'cost_critic_state_dict': self.cost_critic.state_dict(),
                    'cost_critic_target_state_dict': self.cost_critic_target.state_dict(),
                    'cost_critic_optimizer_state_dict': self.cost_critic_optimizer.state_dict(),
                    'cost_value_state_dict': self.cost_value.state_dict()}, file_name)


    def load_model(self, file_name, device_idx=0):
        logger.info(f'Loading models from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.ema_model.load_state_dict(checkpoint['ema_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

    def load_critic(self, file_name, device_idx=0):
        logger.info(f'Loading critics from {file_name}')
        if file_name is not None:
            if device_idx == -1:
                checkpoint = torch.load(file_name, map_location=f'cpu')
            else:
                checkpoint = torch.load(file_name, map_location=f'cuda:{device_idx}')
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.cost_critic.load_state_dict(checkpoint['cost_critic_state_dict'])
            self.cost_critic_target.load_state_dict(checkpoint['cost_critic_target_state_dict'])
            self.cost_value.load_state_dict(checkpoint['cost_value_state_dict'])

    def get_cost_q(self, observation, action):
        _, _, q1_list, q2_list = self.cost_critic.predict(observation, action)
        return q1_list, q2_list

    def get_reward_q(self, observation, action):
        _, _, q1_list, q2_list = self.critic.predict(observation, action)
        return q1_list, q2_list

    def get_vr(self, observations):
        v_list = self.value(observations)
        v = torch.min(torch.stack(v_list), dim=0).values
        return v

    def get_vc(self, observations):
        v_list = self.cost_value(observations)
        v = torch.max(torch.stack(v_list), dim=0).values
        return v
