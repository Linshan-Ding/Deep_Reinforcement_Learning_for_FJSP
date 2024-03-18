from collections import Counter
from torch import nn
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from agents.Base_Agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from utilities.data_structures.Config import Config
from environments.SO_SFJSP import SO_SFJSP_Environment
from visdom import Visdom
from utilities.Utility_Class import AddData

# 训练结果数据保存位置
path_file_name = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/results/DDQN/DDQN_training.csv'
add_data_object = AddData(path_file_name)
add_data_object.add_data(['epoch', 'makespan'])
# 监控训练过程
vis = Visdom()
window_1 = 'completion_time_DDQN'
title_1 = window_1
vis.line(X=[0], Y=[22500], win=window_1, opts=dict(title=title_1, xlabel='epoch', ylable='completion_time', ytickmin=200, font=dict(family='Times New Roman')))


class ActorNet(nn.Module):
    """演员策略网络"""
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(ActorNet, self).__init__()
        # 定义机器策略网络输入层
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.ReLU()])
        # 定义机器策略网络隐藏层
        for i in range(hidden_layer - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU())
        # 定义机器策略网络输出层
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = F.softmax(x, dim=-1)
        return x


class ExplorationStrategy:
    def __init__(self, start_epsilon, min_epsilon, total_episodes):
        self.epsilon = start_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = (start_epsilon - min_epsilon) / total_episodes

    def get_action(self, action_values, turn_off_exploration=False):
        # 如果关闭探索，直接返回最高价值的动作
        if turn_off_exploration:
            self.epsilon = self.min_epsilon

        # 根据当前周期数更新epsilon值，使其线性衰减
        self.epsilon = max(self.min_epsilon, self.epsilon - self.decay_rate)

        # 以epsilon的概率随机选择动作，否则选择最优动作
        if random.random() < self.epsilon:
            return np.random.randint(action_values.size(-1))  # 使用 PyTorch 的 size 方法
        else:
            return torch.argmax(action_values).item()  # 使用 PyTorch 的 argmax


class DDQN(Base_Agent, Config):
    """A deep Q learning agent"""
    def __init__(self):
        Base_Agent.__init__(self)  # 继承基础智能体类
        Config.__init__(self)  # 继承算法超参数类
        self.config = Config()  # 算法控制参数
        self.agent = "DDQN"
        self.hyper_parameters = self.hyper_parameters[self.agent]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 采用GPU训练
        self.memory = Replay_Buffer(self.hyper_parameters["buffer_size"], self.hyper_parameters["batch_size"], self.device)
        # 环境状态维度参数
        self.state_size = 18
        self.action_size = 20
        self.weight_vector_size = 0
        self.actor_input_size = self.state_size
        self.q_network_local = ActorNet(input_size=self.actor_input_size, hidden_size=200, hidden_layer=3, output_size=self.action_size).to(self.device)  # 构建Q神经网络
        self.q_network_optimizer = optim.Adam(self.q_network_local.parameters(), lr=self.hyper_parameters["learning_rate"], eps=1e-4)
        self.q_network_target = ActorNet(input_size=self.actor_input_size, hidden_size=200, hidden_layer=3, output_size=self.action_size).to(self.device)  # 构建Q神经网络
        Base_Agent.copy_model_over(from_model=self.q_network_local, to_model=self.q_network_target)
        # 探索策略
        self.turn_off_exploration = False  # 打开环境探索功能
        self.exploration_strategy = ExplorationStrategy(start_epsilon=1.0, min_epsilon=0.01, total_episodes=self.hyper_parameters["num_episodes_to_run"])
        # 定义测试环境
        self.path = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/data/DDQN'  # 测试算例的存储位置
        self.file_name = 'P51'  # 测试算例的文件夹名字
        self.environment_test = SO_SFJSP_Environment(use_instance=False, path=self.path, file_name=self.file_name)  # 测试环境
        self.environment = None
        self.global_step_number = 0
        self.completed_time = float('inf')

    def generated_new_environment(self):
        """返回新环境对象"""
        DDT = random.uniform(0.5, 1.5)
        M = random.randint(3, 8)
        S = 1
        return SO_SFJSP_Environment(use_instance=True, DDT=DDT, M=M, S=S)

    def step(self):
        """环境运行一个时间步"""
        self.environment = self.generated_new_environment()
        # 重置环境
        self.state = self.environment.reset()  # 重置环境到初始状态
        self.done = False
        while not self.done:
            self.action = self.pick_action(self.turn_off_exploration)  # 选择一个动作
            self.next_state, self.reward, self.done = self.environment.step(self.action)  # 执行一个动作
            self.save_experience(self.memory)  # 存储一条经验|内存中添加经验
            self.state = self.next_state  # 更新下一个状态为当前状态
            self.global_step_number += 1  # 总的步数

        # 更新网络参数
        if self.time_for_q_network_to_learn():
            for _ in range(self.hyper_parameters["learning_iterations"]):  # 每个批次学习迭代数量
                self.learn()

        # 存储测试数据
        self.step_test()
        vis.line(X=[self.episode_number], Y=[self.environment_test.completion_time], win=window_1, update='append')
        add_data_object.add_data([self.episode_number, self.environment_test.completion_time])  # 保存数据
        self.episode_number += 1

        # 保存截止目前性能最优的模型
        if self.environment_test.completion_time < self.completed_time:
            self.completed_time = self.environment_test.completion_time
            self.save_policy_network()
            print('保存模型')

    def save_policy_network(self):
        """保存策略网络"""
        file_path = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/agents/DDQN/'
        torch.save(self.q_network_local.state_dict(), file_path + 'ddqn.path')

    def step_test(self):
        """测试算例"""
        # 重置环境
        self.state = self.environment_test.reset()  # 重置环境到初始状态
        self.done = False
        while not self.done:
            self.action = self.pick_action(True)  # 选择一个动作
            self.next_state, self.reward, self.done = self.environment_test.step(self.action)  # 执行一个动作
            self.state = self.next_state  # 更新下一个状态为当前状态

    def pick_action(self, turn_off_exploration):
        """基于主网络采样动作"""
        state = self.state
        if isinstance(state, np.int64) or isinstance(state, int):
            state = np.array([state])
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) < 2:
            state = state.unsqueeze(0)
        self.q_network_local.eval()  # puts network in evaluation mode
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()  # puts network back in training mode

        # 使用探索策略的示例
        action = self.exploration_strategy.get_action(action_values, turn_off_exploration)

        return action

    def learn(self, experiences=None):
        """Runs a learning iteration for the Q network"""
        # 采样经验
        if experiences is None:
            states, actions, rewards, next_states, dones = self.sample_experiences()  # Sample experiences
        else:
            states, actions, rewards, next_states, dones = experiences
        # 计算损失
        loss = self.compute_loss(states, next_states, rewards, actions, dones)
        self.take_optimisation_step(self.q_network_optimizer, self.q_network_local, loss, self.hyper_parameters["gradient_clipping_norm"])
        # 软更新目标网络
        self.soft_update_of_target_network(self.q_network_local, self.q_network_target, self.hyper_parameters["tau"])

    def compute_loss(self, states, next_states, rewards, actions, dones):
        """计算损失函数"""
        with torch.no_grad():
            Q_targets = self.compute_q_targets(next_states, rewards, dones)
        Q_expected = self.compute_expected_q_values(states, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def compute_q_targets(self, next_states, rewards, dones):
        """计算目标网络预测值"""
        Q_targets_next = self.compute_q_values_for_next_states(next_states)
        Q_targets = self.compute_q_values_for_current_states(rewards, Q_targets_next, dones)
        return Q_targets

    def compute_q_values_for_next_states(self, next_states):
        """基于主网络计算各动作概率"""
        max_action_indexes = self.q_network_local(next_states).detach().argmax(1)
        Q_targets_next = self.q_network_target(next_states).gather(1, max_action_indexes.unsqueeze(1))
        return Q_targets_next

    def compute_q_values_for_current_states(self, rewards, Q_targets_next, dones):
        """基于目标网络计算期望回报"""
        Q_targets_current = rewards + (self.hyper_parameters["discount_rate"] * Q_targets_next * (1 - dones))
        return Q_targets_current

    def compute_expected_q_values(self, states, actions):
        """基于主网络计算期望回报"""
        Q_expected = self.q_network_local(states).gather(1, actions.long())  # must convert actions to long so can be used as index
        return Q_expected

    def time_for_q_network_to_learn(self):
        """学习出发控制"""
        return self.right_amount_of_steps_taken() and self.enough_experiences_to_learn_from(self.memory, self.hyper_parameters["batch_size"])

    def right_amount_of_steps_taken(self):
        """
        Returns boolean indicating whether enough steps have been taken for learning to begin
        """
        return self.global_step_number % self.hyper_parameters["update_every_n_steps"] == 0

    def sample_experiences(self):
        """随机批量采样数据"""
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        return states, actions, rewards, next_states, dones


# 测试算法
if __name__ == '__main__':
    ddqn_object = DDQN()
    for epoch in range(ddqn_object.hyper_parameters["num_episodes_to_run"]):
        ddqn_object.step()
