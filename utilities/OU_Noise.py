import numpy as np
import random
import copy


class OU_Noise():
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = None
        self.reset()

    def reset(self):
        """将内部状态(= noise)重置为均值(mu)"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """更新内部状态并将其作为噪声样本返回"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.array([np.random.normal() for _ in range(len(self.state))])
        self.state += dx
        return self.state