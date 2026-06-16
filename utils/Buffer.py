#%% D4RL Buffer
import numpy as np
import torch

from tqdm import trange
from utils import Batch_Class
from loguru import logger

def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """
    Calculate the discounted cumulative sum of x (can be rewards or costs).
    """
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum

class data_buffer(object):
    def __init__(self, d4rl_data, device, reward_tune='no', state_norm=False, act_noise_std=0, obs_noise_std=0):
        if "next_observations" in d4rl_data.keys():
            buffer = Batch_Class.SampleBatch(
                obs=d4rl_data['observations'],
                obs_next=d4rl_data['next_observations'],
                act=d4rl_data['actions'],
                rew=np.expand_dims(np.squeeze(d4rl_data['rewards']), 1),
                cos=np.expand_dims(np.squeeze(d4rl_data['costs']), 1),
                done=np.expand_dims(np.squeeze(d4rl_data['terminals']), 1),#|d4rl_data['timeouts']
                timeout=np.expand_dims(np.squeeze(d4rl_data['timeouts']), 1),
            )
        else:
            buffer = Batch_Class.SampleBatch(
                obs=d4rl_data['observations'],
                obs_next=d4rl_data['observations'][1:],
                act=d4rl_data['actions'],
                rew=np.expand_dims(np.squeeze(d4rl_data['rewards']), 1),
                cos=np.expand_dims(np.squeeze(d4rl_data['costs']), 1),
                done=np.expand_dims(np.squeeze(d4rl_data['terminals']), 1),
                timeout=np.expand_dims(np.squeeze(d4rl_data['timeouts']), 1),
            )

        self.act_dim = buffer.act.shape[1]
        self.obs_dim = buffer.obs.shape[1]
        self.size = buffer.obs.shape[0]
        self.max_action = 1.0
        self.device = device

        if act_noise_std != 0:
            buffer.act =buffer.act + act_noise_std * np.random.random([self.size, self.act_dim])
            logger.info("Noises have been added to Action Buffer")
        if obs_noise_std != 0:
            buffer.obs = buffer.obs + obs_noise_std * np.random.random([self.size,self.obs_dim])
            buffer.obs_next = buffer.obs_next + obs_noise_std * np.random.random([self.size,self.obs_dim])
            logger.info("Noises have been added to Obs and Obs_next Buffer")

        self.obs = torch.FloatTensor(buffer.obs)
        self.obs_next = torch.FloatTensor(buffer.obs_next)
        if state_norm:
            self.obs_mean = torch.mean(self.obs, dim=0)
            self.obs_std = torch.std(self.obs, dim=0) + 1e-4
            self.obs = (self.obs - self.obs_mean) / self.obs_std
            self.obs_next = (self.obs_next - self.obs_mean) / self.obs_std

        self.act = torch.FloatTensor(buffer.act)
        not_done = np.ones([self.size,1])
        self.not_done = torch.FloatTensor(not_done-buffer.done)
        self.timeouts = torch.FloatTensor(buffer.timeout)
        reward = self.Tune_reward(buffer, reward_tune)
        self.reward = torch.FloatTensor(reward)
        self.cost = torch.FloatTensor(buffer.cos)

        ##########################################################
        # get the indices of the transitions after terminal states or timeouts
        done_idx = np.where(buffer.done|buffer.timeout)[0]
        _cost_thresholds = np.zeros_like(buffer.cos)
        _next_cost_thresholds = np.zeros_like(buffer.cos)

        # compute episode returns
        for i in trange(done_idx.shape[0], desc="Processing trajectories"):
            start = 0 if i == 0 else done_idx[i - 1] + 1
            end = done_idx[i] + 1

            cost_thresholds = discounted_cumsum(buffer.cos[start:end], gamma=1.0)
            next_cost_thresholds = np.zeros_like(cost_thresholds)
            next_cost_thresholds[:-1] = cost_thresholds[1:]

            _cost_thresholds[start:end] = cost_thresholds
            _next_cost_thresholds[start:end] = next_cost_thresholds

        self.obs_min = np.min(buffer.obs, axis=0)
        self.obs_max = np.max(buffer.obs, axis=0)

        self._cost_thre = torch.FloatTensor(_cost_thresholds)
        self._next_cost_thre = torch.FloatTensor(_next_cost_thresholds)

        # Precompute valid starts for trajectory sampling
        self._precompute_valid_starts(buffer)

    def _precompute_valid_starts(self, buffer):
        """Precompute valid starting positions for different trajectory lengths"""
        # Find episode boundaries (where done=True)
        done_indices = np.where((buffer.timeout|buffer.done).squeeze())[0]

        # Store valid starts for different k values (up to reasonable limit)
        self.valid_starts_cache = {}
        max_k = min(50, self.size // 2)  # Cache up to 100 or half buffer size

        for k in range(1, max_k + 1):
            valid_starts = []
            episode_start = 0

            for done_idx in done_indices:
                # Check positions in current episode that allow k consecutive steps
                episode_end = done_idx
                for start in range(episode_start, episode_end - k + 2):
                    if start + k <= episode_end + 1:
                        valid_starts.append(start)
                episode_start = episode_end + 1

            # Handle the last episode if it doesn't end with done=True
            if episode_start < self.size:
                for start in range(episode_start, self.size - k + 1):
                    valid_starts.append(start)

            self.valid_starts_cache[k] = torch.tensor(valid_starts) if valid_starts else torch.tensor([])

    def sample(self, batch_size, sample_setting=None):
        ind = torch.randint(0,self.size-1, size=(batch_size,))
        if sample_setting is None:
            return(
                self.obs[ind].to(self.device),
                self.obs_next[ind].to(self.device),
                self.act[ind].to(self.device),
                self.reward[ind].to(self.device),
                self.cost[ind].to(self.device),
                self.not_done[ind].to(self.device)
            )
        elif sample_setting == "next_action":
            return (
                self.obs[ind].to(self.device),
                self.obs_next[ind].to(self.device),
                self.act[ind].to(self.device),
                self.act[ind+1].to(self.device),
                self.reward[ind].to(self.device),
                self.cost[ind].to(self.device),
                self.not_done[ind].to(self.device)
            )
        elif sample_setting == "cost_threshold":
            return (
                self.obs[ind].to(self.device),
                self.obs_next[ind].to(self.device),
                self.act[ind].to(self.device),
                self.reward[ind].to(self.device),
                self.cost[ind].to(self.device),
                self.not_done[ind].to(self.device),
                self._cost_thre[ind].to(self.device),
                self._next_cost_thre[ind].to(self.device)
            )
        else:
            raise ValueError(f"Please Check the sample setting, '{sample_setting}' does not exist")

    def sample_k_step_trajectory_(self, k, batch_size=256):

        # 确保k小于最短轨迹长度
        max_possible_start = self.size - k

        # 如果数据集太小无法采样k步
        if max_possible_start <= 0:
            raise ValueError(f"Cannot sample {k}-step trajectory from buffer of size {self.size}")

        # 随机选择起始点
        start_idx = torch.randint(0, max_possible_start, size=(batch_size,))

        indices = (start_idx.unsqueeze(1) + torch.arange(k).unsqueeze(0)).reshape(-1)
        # 获取连续k步的索引

        return (
            self.obs[indices].to(self.device).reshape(batch_size, k, -1),  # 状态序列
            self.obs_next[indices].to(self.device).reshape(batch_size, k, -1),  # 下一个状态序列
            self.act[indices].to(self.device).reshape(batch_size, k, -1),  # 动作序列
            self.reward[indices].to(self.device).reshape(batch_size, k, -1),  # 奖励序列
            self.cost[indices].to(self.device).reshape(batch_size, k, -1),  # 代价序列
            self.not_done[indices].to(self.device).reshape(batch_size, k, -1)  # 终止标志序列
        )

    def sample_k_step_trajectory(self, k, batch_size=256):
        # Use precomputed valid starts if available
        if k in self.valid_starts_cache:
            valid_starts = self.valid_starts_cache[k]
        else:
            # Fallback to computing on-demand for very large k values
            done_indices = torch.where(self.not_done.squeeze() == 0)[0].cpu().numpy()
            valid_starts = []
            episode_start = 0

            for done_idx in done_indices:
                episode_end = done_idx.item()
                for start in range(episode_start, episode_end - k + 2):
                    if start + k <= episode_end + 1:
                        valid_starts.append(start)
                episode_start = episode_end + 1

            if episode_start < self.size:
                for start in range(episode_start, self.size - k + 1):
                    valid_starts.append(start)

            valid_starts = torch.tensor(valid_starts)

        if len(valid_starts) == 0:
            raise ValueError(f"No valid {k}-step trajectories found that don't cross episode boundaries")

        # Sample from valid starting positions
        sampled_indices = torch.randint(0, len(valid_starts), size=(batch_size,))
        start_idx = valid_starts[sampled_indices]

        # Generate k consecutive indices for each sampled trajectory
        indices = (start_idx.unsqueeze(1) + torch.arange(k).unsqueeze(0)).reshape(-1)

        return (
            self.obs[indices].to(self.device).reshape(batch_size, k, -1),
            self.obs_next[indices].to(self.device).reshape(batch_size, k, -1),
            self.act[indices].to(self.device).reshape(batch_size, k, -1),
            self.reward[indices].to(self.device).reshape(batch_size, k, -1),
            self.cost[indices].to(self.device).reshape(batch_size, k, -1),
            self.not_done[indices].to(self.device).reshape(batch_size, k, -1)
        )

    def Tune_reward(self, buffer, reward_tune):
        original_reward = buffer.rew
        if reward_tune == "no":
            reward = original_reward
        elif reward_tune == "normalize":
            reward = (original_reward - original_reward.mean())/original_reward.std()
        elif reward_tune == "iql_antmaze":
            reward = original_reward - 1.0
        elif reward_tune == "iql_locomotion":
            reward = self.iql_normalize(original_reward)
        elif reward_tune == "cql_antmaze":
            reward = (original_reward - 0.5) * 4.0
        elif reward_tune == "antmaze":
            reward = (original_reward - 0.25) * 2.0
        elif reward_tune == 'punish':
            idx = np.where(buffer.done)
            idx = idx[0]
            reward = original_reward
            reward[idx] -= 100
            # idx -= 1
            # reward[idx] -= 40
            # idx -= 1
            # reward[idx] -= 30
            # idx -= 1
            # reward[idx] -= 20
            # idx -= 1
            # reward[idx] -= 10
        else:
            raise ValueError(f"Please Check the reward tuning method, '{reward_tune}' does not exist")
        return reward

    def iql_normalize(self, reward):
        trajs_rt = []
        episode_return = 0.0
        for i in range(len(reward)):
            episode_return += reward[i]
            if not self.not_done[i]:
                trajs_rt.append(episode_return)
                episode_return = 0.0
        rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
        reward /= (rt_max-rt_min)
        reward *= 1000
        return reward
