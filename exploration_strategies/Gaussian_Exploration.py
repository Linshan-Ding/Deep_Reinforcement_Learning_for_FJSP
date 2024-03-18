from exploration_strategies.Base_Exploration_Strategy import Base_Exploration_Strategy
import torch
from torch.distributions.normal import Normal


class Gaussian_Exploration(Base_Exploration_Strategy):
    """高斯噪声探测策略"""
    def __init__(self, config, algorithm, device=None):
        super().__init__(config)
        self.device = device
        self.action_noise_std = self.config.hyper_parameters[algorithm]["action_noise_std"]
        self.action_noise_distribution = Normal(torch.Tensor([0.0]), torch.Tensor([self.action_noise_std]))
        self.action_noise_clipping_range = self.config.hyper_parameters[algorithm]["action_noise_clipping_range"]

    def perturb_action_for_exploration_purposes(self, action_info):
        """干扰代理的行动以鼓励探索"""
        action = action_info["action"]
        action_noise = self.action_noise_distribution.sample(sample_shape=action.shape)
        action_noise = action_noise.squeeze(-1)
        clipped_action_noise = torch.clamp(action_noise, min=-self.action_noise_clipping_range, max=self.action_noise_clipping_range)
        action += clipped_action_noise.to(self.device)
        return action

    def add_exploration_rewards(self, reward_info):
        """鼓励探索的内在奖励"""
        raise ValueError("Must be implemented")

    def reset(self):
        """Resets the noise process"""
        pass

