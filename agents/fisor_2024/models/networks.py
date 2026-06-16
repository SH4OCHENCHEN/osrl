import torch
import torch.nn as nn
import numpy as np



def mlp_(sizes, activation, output_activation=nn.Identity):
    """
    Creates a multi-layer perceptron with the specified sizes and activations.

    Args:
        sizes (list): A list of integers specifying the size of each layer in the MLP.
        activation (nn.Module): The activation function to use for all layers except the output layer.
        output_activation (nn.Module): The activation function to use for the output layer. Defaults to nn.Identity.

    Returns:
        nn.Sequential: A PyTorch Sequential model representing the MLP.
    """

    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layer = nn.Linear(sizes[j], sizes[j + 1])
        layers += [layer, act()]
    return nn.Sequential(*layers)


def mlp(sizes, activation, output_activation=nn.Identity, layernorm=True, dropout=0.0):
    """
    Creates a multi-layer perceptron with the specified sizes and activations,
    optionally adding LayerNorm and Dropout after each hidden layer.

    Args:
        sizes (list): Layer sizes.
        activation (nn.Module): Activation for hidden layers.
        output_activation (nn.Module): Activation for output layer.
        layernorm (bool): Whether to add LayerNorm after each hidden layer.
        dropout (float): Dropout probability after each activation.

    Returns:
        nn.Sequential: The constructed MLP.
    """
    layers = []
    for j in range(len(sizes) - 1):
        layer = nn.Linear(sizes[j], sizes[j + 1])
        if j < len(sizes) - 2:
            layer_ = [layer]
            if layernorm:
                layer_.append(nn.LayerNorm(sizes[j + 1]))
            layer_.append(activation())
            if dropout > 0.0:
                layer_.append(nn.Dropout(dropout))
        else:
            layer_ = [layer, output_activation()]
        layers += layer_
    return nn.Sequential(*layers)


class FeaturesTokenizer(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(FeaturesTokenizer, self).__init__()
        """
        TOKENIZE THE STATES' FEATURES INTO EMBEDDING.
        """

        self.weights = nn.Parameter(torch.randn(input_dim, embed_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(input_dim, embed_dim), requires_grad=True)

    def forward(self, x):
        """
        X SHOULD BE A BATCH_SIZE X STATE_DIM TENSOR
        """
        x_proj = x.unsqueeze(-1) * self.weights + self.bias

        return x_proj


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_head):

        super(TransformerBlock, self).__init__()

        """
        INPUT IS A BATCH_SIZE X SEQUENCE_LENGTH X EMBEDDING_DIM TENSOR
        THIS MODULE DO TRANSFORMER LAYER OPERATION TO IT
        WHILE WE USE DIT ADAPTIVE LAYERNORM
        """

        self.embed_dim = embed_dim
        self.num_head = num_head
        self.hidden_dim = hidden_dim

        self.Q_linear = nn.Linear(embed_dim, embed_dim*num_head)
        self.K_linear = nn.Linear(embed_dim, embed_dim*num_head)
        self.V_linear = nn.Linear(embed_dim, embed_dim*num_head)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.Mish(),
            # nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )

        self.feed_forward_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)
        self.multihead_attention_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

        self.attention_output_layer = nn.Linear(embed_dim*num_head, embed_dim)

    def forward(self, x):
        """
        X IS A BATCH_SIZE X SEQUENCE_LENGTH X EMBEDDING_DIM TENSOR, THE NOISED STATE
        """

        # DO MULTI HEAD ATTENTION, USE PRENORM STYLE
        _x = self.multihead_attention_norm(x)

        query = self.Q_linear(_x)
        key = self.K_linear(_x).permute(0, 2, 1)
        value = self.V_linear(_x)

        multihead_attention = []
        for i in range(self.num_head):
            output = torch.matmul(
                query[:,:,i*self.embed_dim:(i+1)*self.embed_dim],
                key[:,i*self.embed_dim:(i+1)*self.embed_dim,:]
            ) / (float(self.embed_dim) ** 0.5)
            out_map = nn.Softmax(dim=-1)(output)
            output = torch.matmul(out_map, value[:,:,i*self.embed_dim:(i+1)*self.embed_dim])
            multihead_attention.append(output)

        multihead_attention_concat = torch.cat(multihead_attention, dim=-1)
        output = self.attention_output_layer(multihead_attention_concat)

        # SKIP CONNECT
        x = x + output

        # DO THE FEEDFORWARD
        _x = self.feed_forward_norm(x)
        ffn_output = self.feed_forward(_x)

        # SKIP CONNECT
        x = x + ffn_output

        return x



class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_head):

        super(MultiHeadAttentionBlock, self).__init__()

        """
        INPUT IS A BATCH_SIZE X SEQUENCE_LENGTH X EMBEDDING_DIM TENSOR
        THIS MODULE DO MULTIHEAD ATTENTION OPERATION TO INPUT
        THE OUTPUT IS A BATCH_SIZE X SEQUENCE_LENGTH X EMBEDDING_DIM TENSOR
        """

        self.embed_dim = embed_dim
        self.num_head = num_head

        self.Q_linear = nn.Linear(embed_dim, embed_dim*num_head)
        self.K_linear = nn.Linear(embed_dim, embed_dim*num_head)
        self.V_linear = nn.Linear(embed_dim, embed_dim*num_head)

        self.multihead_attention_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

        self.attention_output_layer = nn.Linear(embed_dim*num_head, embed_dim)

    def forward(self, x):
        """
        X IS A BATCH_SIZE X SEQUENCE_LENGTH X EMBEDDING_DIM TENSOR, THE NOISED STATE
        """

        # DO MULTI HEAD ATTENTION, USE PRENORM STYLE
        _x = self.multihead_attention_norm(x)

        query = self.Q_linear(_x)
        key = self.K_linear(_x).permute(0, 2, 1)
        value = self.V_linear(_x)

        multihead_attention = []
        for i in range(self.num_head):
            output = torch.matmul(
                query[:,:,i*self.embed_dim:(i+1)*self.embed_dim],
                key[:,i*self.embed_dim:(i+1)*self.embed_dim,:]
            ) / (float(self.embed_dim) ** 0.5)
            out_map = nn.Softmax(dim=-1)(output)
            output = torch.matmul(out_map, value[:,:,i*self.embed_dim:(i+1)*self.embed_dim])
            multihead_attention.append(output)

        multihead_attention_concat = torch.cat(multihead_attention, dim=-1)
        output = self.attention_output_layer(multihead_attention_concat)

        # SKIP CONNECT
        x = x + output

        return x

class Self_Attention(nn.Module):
    def __init__(self, input_dim, out_dim, embed_dim, hidden_dim, head_num, device='cuda:0'):
        super(Self_Attention, self).__init__()
        self.device = device

        self.q_dim = input_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.head_num = head_num

        self.Attention = nn.ModuleList([
            Attention_Layer(self.embed_dim, self.embed_dim, True)
            for i in range(head_num)
        ])

        self.output_linear = nn.Linear(self.head_num * self.embed_dim, self.embed_dim)
        self.out_norm = nn.LayerNorm(self.embed_dim)
        self.feed_forward = nn.Sequential(
            torch.nn.Linear(self.embed_dim, self.hidden_dim*2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_dim*2, self.embed_dim)
        )
        self.feed_forward_norm = nn.LayerNorm(self.embed_dim)

        self.attention_out_layer = nn.Linear(self.embed_dim * self.q_dim, self.out_dim)

        self.W = nn.Parameter(torch.randn(self.embed_dim // 2) * 30., requires_grad=True)
        self.act = nn.Mish()

    def encoder_for_info(self, input):
        x_proj = input * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(2)
        Info = self.encoder_for_info(inputs)
        Multi_head = [Atn(Info)[0] for Atn in self.Attention]
        output = torch.cat(Multi_head, dim=2)
        output = self.output_linear(output)
        output = self.out_norm(output + Info)
        output = self.feed_forward(output) + output
        output = self.feed_forward_norm(output)

        output = torch.flatten(output, start_dim=1)
        output = self.attention_out_layer(self.act(output))

        return output

    def get_attention_map(self, inputs):
        inputs = inputs.unsqueeze(2)
        Info = self.encoder_for_info(inputs)
        attention_maps = [Atn(Info)[1] for Atn in self.Attention]
        return attention_maps


class Attention_Layer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, self_attention=True):
        super(Attention_Layer, self).__init__()
        self.self_attention = self_attention

        self.Q_linear = nn.Linear(embed_dim, hidden_dim)
        self.K_linear = nn.Linear(embed_dim, hidden_dim)
        self.V_linear = nn.Linear(embed_dim, hidden_dim)
        self.W_linear = nn.Parameter(torch.randn((hidden_dim, hidden_dim)), requires_grad=True)

    def forward(self, input, input_KV=None):
        if not self.self_attention and input_KV is None:
            raise ValueError('For Cross Attention Network, information of Key and Value must be provided.')
        if self.self_attention and input_KV is not None:
            raise ValueError('For Self Attention Network, information of Q, K, V must be the same')
        if self.self_attention:
            Query = self.Q_linear(input)
            Key = self.K_linear(input).permute(0, 2, 1)
            Value = self.V_linear(input)
        else:
            Query = self.Q_linear(input)
            Key = self.K_linear(input_KV).permute(0, 2, 1)
            Value = self.V_linear(input_KV)

        output = torch.matmul(torch.matmul(Query, self.W_linear), Key) / (float(Value.size(-1)) ** 0.5)
        out_map = nn.Softmax(dim=-1)(output)
        output = torch.matmul(out_map, Value)
        return output, out_map


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30., device='cuda:0'):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2, device=device) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class DoubleQCritic(nn.Module):
    '''
    double q net using latent attention encoder.
    '''

    def __init__(self, state_dim, action_dim, hidden_dim, head_num, embed_dim, state_latent_dim, device='cuda:0'):
        super(DoubleQCritic, self).__init__()
        self.state_encoder = (
            Self_Attention(state_dim, state_latent_dim, embed_dim, hidden_dim, head_num, device=device))

        self.critic1 = nn.Sequential(
            nn.Linear(state_latent_dim + action_dim, hidden_dim*2),
            nn.Mish(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.Mish(),
            nn.Linear(hidden_dim*2, 1),
        )

        self.critic2 = nn.Sequential(
            nn.Linear(state_latent_dim + action_dim, hidden_dim*2),
            nn.Mish(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.Mish(),
            nn.Linear(hidden_dim*2, 1),
        )

    def forward(self, state, action):
        latent_state = self.state_encoder(state)
        inputs = torch.cat([latent_state, action], dim=-1)

        q1 = self.critic1(inputs)
        q2 = self.critic2(inputs)
        return q1, q2


    def q1(self, state, action):
        latent_state = self.state_encoder(state)
        inputs = torch.cat([latent_state, action], dim=-1)

        q1 = self.critic1(inputs)
        return q1

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

    def q_max(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.max(q1, q2)

    def get_attention_map(self, inputs):
        return self.state_encoder.get_attention_map(inputs)

    def predict(self, obs, act):
        q1_list, q2_list = self.forward(obs, act)
        q1_list = [torch.squeeze(q1_list, -1)]
        q2_list = [torch.squeeze(q2_list, -1)]
        qs1, qs2 = torch.vstack(q1_list), torch.vstack(q2_list)
        qs1_min, qs2_min = torch.min(qs1, dim=0).values, torch.min(qs2, dim=0).values
        return qs1_min, qs2_min, q1_list, q2_list


class DoubleQAttentionCritic(nn.Module):
    '''
    double q net using latent attention encoder.
    '''

    def __init__(self, state_dim, action_dim, hidden_dim, head_num, embed_dim, state_latent_dim, device='cuda:0'):
        super(DoubleQAttentionCritic, self).__init__()

        self.input_socket = FeaturesTokenizer(state_dim, embed_dim)

        self.attention_socket = TransformerBlock(embed_dim, hidden_dim, head_num)

        self.attention_final_layer = nn.Linear(embed_dim * state_dim, state_latent_dim)

        self.critic1 = nn.Sequential(
            nn.Linear(state_latent_dim + action_dim, hidden_dim*2),
            nn.Mish(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.Mish(),
            nn.Linear(hidden_dim*2, 1),
        )

        self.critic2 = nn.Sequential(
            nn.Linear(state_latent_dim + action_dim, hidden_dim*2),
            nn.Mish(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.Mish(),
            nn.Linear(hidden_dim*2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # ZERO OUT THE OUTPUT
        nn.init.constant_(self.critic1[-1].weight, 0)
        nn.init.constant_(self.critic1[-1].bias, 0)
        nn.init.constant_(self.critic2[-1].weight, 0)
        nn.init.constant_(self.critic2[-1].bias, 0)

    def forward(self, state, action):

        embedded_state = self.input_socket(state)

        attention_out = self.attention_socket(embedded_state)
        _attention_out = torch.flatten(attention_out, start_dim=1)
        latent_state = self.attention_final_layer(_attention_out)

        inputs = torch.cat([latent_state, action], dim=-1)

        q1 = self.critic1(inputs)
        q2 = self.critic2(inputs)
        return q1, q2

    def q1(self, state, action):
        q1, q2 = self.forward(state, action)
        return q1

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

    def q_max(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.max(q1, q2)

    def get_attention_map(self, inputs):
        return self.state_encoder1.get_attention_map(inputs)

    def predict(self, obs, act):
        q1_list, q2_list = self.forward(obs, act)
        q1_list = [torch.squeeze(q1_list, -1)]
        q2_list = [torch.squeeze(q2_list, -1)]
        qs1, qs2 = torch.vstack(q1_list), torch.vstack(q2_list)
        qs1_min, qs2_min = torch.min(qs1, dim=0).values, torch.min(qs2, dim=0).values
        return qs1_min, qs2_min, q1_list, q2_list


class DoubleQSimplifiedAttentionCritic(nn.Module):
    '''
    double q net using latent attention encoder.
    compared to DoubleQAttentionCritic, we do not use ffn in the attention encoder.
    i.e. directly output the encoded state from the multihead attention output
    '''

    def __init__(self, state_dim, action_dim, hidden_dim, head_num, embed_dim, state_latent_dim, device='cuda:0'):
        super(DoubleQSimplifiedAttentionCritic, self).__init__()

        self.input_socket = FeaturesTokenizer(state_dim, embed_dim)

        self.attention_socket = MultiHeadAttentionBlock(embed_dim, head_num)

        self.attention_final_layer = nn.Linear(embed_dim * state_dim, state_latent_dim)

        self.critic1 = nn.Sequential(
            nn.Linear(state_latent_dim + action_dim, hidden_dim*2),
            nn.Mish(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.Mish(),
            nn.Linear(hidden_dim*2, 1),
        )

        self.critic2 = nn.Sequential(
            nn.Linear(state_latent_dim + action_dim, hidden_dim*2),
            nn.Mish(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.Mish(),
            nn.Linear(hidden_dim*2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # ZERO OUT THE OUTPUT
        nn.init.constant_(self.critic1[-1].weight, 0)
        nn.init.constant_(self.critic1[-1].bias, 0)
        nn.init.constant_(self.critic2[-1].weight, 0)
        nn.init.constant_(self.critic2[-1].bias, 0)

    def forward(self, state, action):

        embedded_state = self.input_socket(state)

        attention_out = self.attention_socket(embedded_state)
        _attention_out = torch.flatten(attention_out, start_dim=1)
        latent_state = self.attention_final_layer(_attention_out)

        inputs = torch.cat([latent_state, action], dim=-1)

        q1 = self.critic1(inputs)
        q2 = self.critic2(inputs)
        return q1, q2

    def q1(self, state, action):
        q1, q2 = self.forward(state, action)
        return q1

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

    def q_max(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.max(q1, q2)

    def get_attention_map(self, inputs):
        return self.state_encoder1.get_attention_map(inputs)

    def predict(self, obs, act):
        q1_list, q2_list = self.forward(obs, act)
        q1_list = [torch.squeeze(q1_list, -1)]
        q2_list = [torch.squeeze(q2_list, -1)]
        qs1, qs2 = torch.vstack(q1_list), torch.vstack(q2_list)
        qs1_min, qs2_min = torch.min(qs1, dim=0).values, torch.min(qs2, dim=0).values
        return qs1_min, qs2_min, q1_list, q2_list


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, head_num, embed_dim, device='cuda:0'):
        super(ValueNet, self).__init__()

        self.input_socket = FeaturesTokenizer(state_dim, embed_dim)

        self.attention_socket = TransformerBlock(embed_dim, hidden_dim, head_num)

        self.attention_final_layer = nn.Linear(embed_dim * state_dim, embed_dim)

        self.value_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        embedded_state = self.input_socket(x)

        attention_out = self.attention_socket(embedded_state)
        _attention_out = torch.flatten(attention_out, start_dim=1)
        x = self.attention_final_layer(_attention_out)

        x = self.value_layer(x)

        return x

class SimAttValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, head_num, embed_dim, device='cuda:0'):
        super(SimAttValueNet, self).__init__()

        self.input_socket = FeaturesTokenizer(state_dim, embed_dim)

        self.attention_socket = MultiHeadAttentionBlock(embed_dim, head_num)

        self.attention_final_layer = nn.Linear(embed_dim * state_dim, embed_dim)

        self.value_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        embedded_state = self.input_socket(x)

        attention_out = self.attention_socket(embedded_state)
        _attention_out = torch.flatten(attention_out, start_dim=1)
        x = self.attention_final_layer(_attention_out)

        x = self.value_layer(x)

        return x


class EnsembleQCritic(nn.Module):
    '''
    An ensemble of Q network to address the overestimation issue.

    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
        num_q (float): The number of Q networks to include in the ensemble.
    '''

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, use_layer_norm=True, num_q=2):
        super().__init__()
        assert num_q >= 1, "num_q param should be greater than 1"

        self.q_nets = nn.ModuleList([
            mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation, layernorm=use_layer_norm)
            for i in range(num_q)
        ])

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, obs, act=None):
        # Squeeze is critical to ensure value has the right shape.
        # Without squeeze, the training stability will be greatly affected!
        # For instance, shape [3] - shape[3,1] = shape [3, 3] instead of shape [3]
        data = obs if act is None else torch.cat([obs, act], dim=-1)
        return [q(data) for q in self.q_nets]
        # return [torch.squeeze(torch.nn.functional.softplus(q(data)), -1) for q in self.q_nets]

    def predict(self, obs, act):
        q_list = self.forward(obs, act)
        qs = torch.stack(q_list)# [num_q, batch_size]
        return torch.min(qs, dim=0).values, q_list
        # return torch.mean(qs, dim=0), q_list


    def loss(self, target, q_list=None):
        losses = [torch.nn.functional.mse_loss(q, target) for q in q_list]
        return sum(losses)


class EnsembleDoubleQCritic(nn.Module):
    '''
    An ensemble of double Q network to address the overestimation issue.

    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
        num_q (float): The number of Q networks to include in the ensemble.
    '''

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, num_q=2):
        super().__init__()
        assert num_q >= 1, "num_q param should be greater than 1"
        self.q1_nets = nn.ModuleList([
            mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU)
            for i in range(num_q)
        ])
        self.q2_nets = nn.ModuleList([
            mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], nn.ReLU)
            for i in range(num_q)
        ])

    def forward(self, obs, act):
        # Squeeze is critical to ensure value has the right shape.
        # Without squeeze, the training stability will be greatly affected!
        # For instance, shape [3] - shape[3,1] = shape [3, 3] instead of shape [3]
        data = torch.cat([obs, act], dim=-1)
        q1 = [q(data) for q in self.q1_nets]
        q2 = [q(data) for q in self.q2_nets]
        return q1, q2

    def predict(self, obs, act):
        q1_list, q2_list = self.forward(obs, act)
        qs1, qs2 = torch.vstack(q1_list), torch.vstack(q2_list)
        # qs = torch.vstack(q_list)  # [num_q, batch_size]
        qs1_min, qs2_min = torch.min(qs1, dim=0).values, torch.min(qs2, dim=0).values
        return qs1_min, qs2_min, q1_list, q2_list

    def loss(self, target, q_list=None):
        losses = [((q - target)**2).mean() for q in q_list]
        return sum(losses)


class EnsembleValue(nn.Module):
    '''
    An ensemble of Value network to address the overestimation issue.

    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
        num_v (float): The number of Value networks to include in the ensemble.
    '''

    def __init__(self, obs_dim, hidden_sizes, activation, use_layer_norm=True, num_v=2):
        super().__init__()
        assert num_v >= 1, "num_q param should be greater than 1"

        self.v_nets = nn.ModuleList([
            mlp([obs_dim] + list(hidden_sizes) + [1], activation, layernorm=use_layer_norm)
            for i in range(num_v)
        ])

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, obs):
        data = obs
        return [v(data) for v in self.v_nets]
        # return [torch.squeeze(torch.nn.functional.softplus(v(data)), -1) for v in self.v_nets]

    def predict(self, obs):
        v_list = self.forward(obs)
        vs = torch.stack(v_list)  # [num_q, batch_size]
        return torch.min(vs, dim=0).values, v_list
        # return torch.mean(qs, dim=0), q_list


    def loss(self, target, v_list=None):
        losses = [torch.nn.functional.mse_loss(v, target) for v in v_list]
        return sum(losses)


class EnsembleTauValue(nn.Module):
    '''
    An ensemble of Value network to address the overestimation issue.

    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
        num_v (float): The number of Value networks to include in the ensemble.
    '''

    def __init__(self, obs_dim, hidden_sizes, activation, num_v=2):
        super().__init__()
        assert num_v >= 1, "num_q param should be greater than 1"

        self.v_nets = nn.ModuleList([
            mlp([obs_dim+64] + list(hidden_sizes) + [1], activation)
            for i in range(num_v)
        ])

        self.tau_mlp = nn.Sequential(
            nn.Linear(1, 256),
            nn.Mish(),
            nn.Linear(256, 64),
        )

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, obs, tau):
        tau_embed = self.tau_mlp(tau)  #
        data = torch.concat([obs, tau_embed], dim=-1)
        return [v(data) for v in self.v_nets]
        # return [torch.squeeze(torch.nn.functional.softplus(v(data)), -1) for v in self.v_nets]

    def predict(self, obs, tau):
        v_list = self.forward(obs, tau)
        vs = torch.stack(v_list)  # [num_q, batch_size]
        return torch.min(vs, dim=0).values, v_list
        # return torch.mean(qs, dim=0), q_list


    def loss(self, target, v_list=None):
        losses = [torch.nn.functional.mse_loss(v, target) for v in v_list]
        return sum(losses)


class EnsembleTauQCritic(nn.Module):
    '''
    An ensemble of Q network to address the overestimation issue.

    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_sizes (List[int]): The sizes of the hidden layers in the neural network.
        activation (Type[nn.Module]): The activation function to use between layers.
        num_q (float): The number of Q networks to include in the ensemble.
    '''

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, num_q=2):
        super().__init__()
        assert num_q >= 1, "num_q param should be greater than 1"

        self.q_nets = nn.ModuleList([
            mlp([obs_dim + act_dim + 64] + list(hidden_sizes) + [1], activation)
            for i in range(num_q)
        ])

        self.tau_mlp = nn.Sequential(
            nn.Linear(1, 256),
            nn.Mish(),
            nn.Linear(256, 64),
        )

        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, obs, act, tau):
        tau_embed = self.tau_mlp(tau)
        data = torch.cat([obs, act, tau_embed], dim=-1)
        return [q(data) for q in self.q_nets]
        # return [torch.squeeze(torch.nn.functional.softplus(q(data)), -1) for q in self.q_nets]

    def predict(self, obs, act, tau):
        q_list = self.forward(obs, act, tau)
        qs = torch.stack(q_list)# [num_q, batch_size]
        return torch.min(qs, dim=0).values, q_list
        # return torch.mean(qs, dim=0), q_list

    def loss(self, target, q_list=None):
        losses = [torch.nn.functional.mse_loss(q, target) for q in q_list]
        return sum(losses)
