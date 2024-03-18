
class Base_Exploration_Strategy(object):
    """
    智能体探索策略的基本抽象类。
    每个探索策略都必须继承这个类，并实现方法摄动_action_for_exploration_purposes和add_exploration_rewards
    """
    def __init__(self, config):
        self.config = config

    def perturb_action_for_exploration_purposes(self, action_info):
        """干扰代理的行动以鼓励探索"""
        raise ValueError("Must be implemented")

    def add_exploration_rewards(self, reward_info):
        """鼓励探索的内在奖励"""
        raise ValueError("Must be implemented")

    def reset(self):
        """重置噪声进程"""
        raise ValueError("Must be implemented")