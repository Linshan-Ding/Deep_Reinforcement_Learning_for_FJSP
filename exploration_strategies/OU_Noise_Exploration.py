from utilities.OU_Noise import OU_Noise
from exploration_strategies.Base_Exploration_Strategy import Base_Exploration_Strategy


class OU_Noise_Exploration(Base_Exploration_Strategy):
    """Ornstein-Uhlenbeck噪声过程探索策略"""
    def __init__(self, config, algorithm):
        super().__init__(config)
        self.noise = OU_Noise(self.config.hyper_parameters[algorithm]["action_size"], self.config.seed, self.config.hyper_parameters[algorithm]["mu"],
                              self.config.hyper_parameters[algorithm]["theta"], self.config.hyper_parameters[algorithm]["sigma"])

    def perturb_action_for_exploration_purposes(self, action_info):
        """干扰代理的行动以鼓励探索"""
        action = action_info["action"]
        action += self.noise.sample()
        return action

    def add_exploration_rewards(self, reward_info):
        """鼓励探索的内在奖励"""
        raise ValueError("Must be implemented")

    def reset(self):
        """重置噪声进程"""
        self.noise.reset()