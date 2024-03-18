"""
actor+critic PPO算法+整周期回报+随机训练实例
"""
import numpy as np
import torch, random, copy, sys
from torch import nn
import torch.nn.functional as F
from torch import optim
from agents.Base_Agent import Base_Agent
from Buffer import Replay_Buffer
from utilities.Utility_Functions import create_actor_distribution
from utilities.data_structures.Config import Config
from environments.MO_FJSSP_discretes import MO_FJSSP_Environment
from visdom import Visdom
from utilities.Utility_Class import AddData
agent_version = '_v6.3'
# 训练结果数据保存位置
path_file_name = 'D:/Python project/Deep_Reinforcement_learning_FJSP/results/MPPPO/training' + agent_version + '.csv'
add_data_object = AddData(path_file_name)
add_data_object.add_data(['epoch', 'makespan_avg', 'tardiness_avg', 'makespan_min', 'tardiness_min'])
# 监控训练过程
vis = Visdom()
window_1 = 'completion_time_actor_critic' + agent_version
title_1 = window_1
vis.line(X=[0], Y=[300], win=window_1, opts=dict(title=title_1, xlabel='epoch', ylable='completion_time', font=dict(family='Times New Roman')))
window_2 = 'total_delay_time_actor_critic' + agent_version
title_2 = window_2
vis.line(X=[0], Y=[10000], win=window_2, opts=dict(title=title_2, xlabel='epoch', ylable='total_delay_time', font=dict(family='Times New Roman')))


class ActorNet(nn.Module):
    """演员策略网络"""
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(ActorNet, self).__init__()
        # 定义机器策略网络输入层
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size), nn.ReLU()])
        # 定义机器策略网络隐藏层
        for i in range(hidden_layer - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        # 定义机器策略网络输出层
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = F.softmax(x, dim=-1)
        return x


class CriticNet(nn.Module):
    """评论家网络"""
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(CriticNet, self).__init__()
        # 定义评论家网络输入层
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size), nn.ReLU()])
        # 定义评论家网络隐藏层
        for i in range(hidden_layer - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        # 定义评论家网络输出层
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PPO(Base_Agent, Config):
    """多策略近端策略优化算法"""
    def __init__(self):
        Base_Agent.__init__(self)
        Config.__init__(self)  # 继承算法超参数类
        self.config = Config()  # 算法控制参数
        self.agent = "MP_PPO"
        self.hyper_parameters = self.hyper_parameters[self.agent]  # 算法控制参数
        self.action_types = "DISCRETE"
        self.episode_states = []  # 状态
        self.episode_actions = []  # 动作
        self.episode_rewards = []  # 回报
        self.epsilon_decay_denominator = self.hyper_parameters["epsilon_decay_rate_denominator"]  # 探索衰变率分母
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 采用GPU训练
        self.actor_number = self.hyper_parameters["actor_number"]  # 策略网络数量[5/9/11]
        self.policy_tuple = tuple(p for p in range(self.actor_number))  # 策略网络编号元组
        self.policy_completion = self.policy_tuple[0]  # 完工时间目标网络编号
        self.policy_tardiness = self.policy_tuple[-1]  # 总延迟时间目标网络编号
        self.policy_weight_tuple = self.policy_tuple[1:-1]  # 两个目标权重都非零的策略网络元组
        # 定义测试环境
        self.path = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/data/MPPPO'  # 测试算例的存储位置
        self.file_name = 'DDT1.0_M15_R10'  # 测试算例的文件夹名字
        self.environment = MO_FJSSP_Environment(use_instance=False, path=self.path, file_name=self.file_name)  # 测试环境
        # 超参数
        self.num_episodes_to_run = self.hyper_parameters["num_episodes_to_run"]  # 训练周期数
        self.learning_rate = self.hyper_parameters["learning_rate"]  # 学习率
        self.discount_rate = self.hyper_parameters["discount_rate"]  # 折扣率
        self.buffer_size = self.hyper_parameters["buffer_size"]  # 回放记忆缓存大小
        self.batch_size = self.hyper_parameters["batch_size"]  # 采样批量
        self.gradient_clipping_norm = self.hyper_parameters["gradient_clipping_norm"]  # 梯度裁剪
        self.learning_iterations_per_round_actor = self.hyper_parameters["learning_iterations_per_round_actor"]  # 每次采样更新，演员连续更新轮数
        self.learning_iterations_per_round_critic = self.hyper_parameters["learning_iterations_per_round_critic"]  # 每次采样更新，评论家连续更新轮数
        # 算法运行参数
        self.global_step_number = 0  # 运行总步数
        self.episode_number = 0  # 更新周期数
        # 环境状态维度参数
        self.state_size = 25
        self.action_size = 18
        self.actor_input_size = self.state_size
        self.critic_input_size = self.state_size
        # 生成均匀分布的actor_number个权重向量(completion, tardiness)
        self.weight_vector_dict = {policy: (1 - 1 / (self.actor_number - 1) * policy, 1 / (self.actor_number - 1) * policy) for policy in self.policy_tuple}
        # 定义actor_number个策略网络和目标策略网络
        self.actor_new_dict = {}  # 策略网络对象字典
        self.actor_old_dict = {}  # 目标策略网络对象字典
        self.actor_optimizer_dict = {}  # 策略网络优化器对象字典
        self.critic_dict = {}  # 评论家网络字典
        self.critic_optimizer_dict = {}  # 评论家网络优化器字典
        self.generated_policy_network()  # 示例化多个：策略网络+目标策略网络+策略网络优化器+评论家网络+评论家网络优化器
        # 存储相关值
        self.actor_old_log_prob = []  # 旧网络的动作概率
        self.discounted_returns = []  # 折扣回报
        self.critic_targets = None  # 评论家输出的目标V值
        self.advantages = None  # 优势函数值
        self.log_prob = None  # 动作概率
        # 初始化当前优化的策略网络对象
        self.actor_new = None  # 策略网络对象字典
        self.actor_old = None  # 目标策略网络对象字典
        self.actor_optimizer = None  # 策略网络优化器对象字典
        # 定义评论家网络+评论家优化器
        self.critic_local = None
        self.critic_optimizer = None
        self.memory = Replay_Buffer(device=self.device)  # 记忆缓存字典
        self.weight_vector = None  # 当前策略网络权重
        self.completion_min = float('inf')
        self.tardiness_min = float('inf')
        self.completion_avg = float('inf')
        self.tardiness_avg = float('inf')

    def generated_policy_network(self):
        """生成多个：策略网络+目标策略网络+优化器+回放缓存"""
        for policy in self.policy_tuple:
            self.actor_new_dict[policy] = ActorNet(input_size=self.actor_input_size, hidden_size=200, hidden_layer=5, output_size=self.action_size).to(self.device)
            self.actor_old_dict[policy] = ActorNet(input_size=self.actor_input_size, hidden_size=200, hidden_layer=5, output_size=self.action_size).to(self.device)
            Base_Agent.copy_model_over_dict(self.actor_new_dict[policy], self.actor_old_dict[policy])
            self.actor_optimizer_dict[policy] = optim.Adam(self.actor_new_dict[policy].parameters(), lr=self.learning_rate, eps=1e-4)
            self.critic_dict[policy] = CriticNet(input_size=self.critic_input_size, hidden_size=200, hidden_layer=3, output_size=1).to(self.device)
            self.critic_optimizer_dict[policy] = optim.Adam(self.critic_dict[policy].parameters(), lr=self.learning_rate, eps=1e-4)

    def generated_new_environment(self):
        """返回新环境对象"""
        DDT = random.uniform(0.5, 1.5)
        M = random.randint(10, 20)
        S = 1
        return MO_FJSSP_Environment(use_instance=True, DDT=DDT, M=M, S=S)

    def run_n_episodes(self):
        """多个策略网络串行运行多个训练周期"""
        policy_objectives_dict = {policy: {"completion": None, "tardiness": None} for policy in self.policy_tuple}  # 测试算例下各策略网络的两个目标值变化曲线
        for epoch in range(self.num_episodes_to_run):
            environment = self.generated_new_environment()  # 随机初始化一个训练实例环境
            _, completion = self.run_one_policy_network(environment=environment, policy_number=self.policy_completion)  # 训练最大完工时间策略网络
            tardiness, _ = self.run_one_policy_network(environment=environment, policy_number=self.policy_tardiness)  # 训练总延迟时间策略网络
            for policy in self.policy_weight_tuple:  # 循环训练多个权重策略网络
                _, _ = self.run_one_policy_network(environment=environment, policy_number=policy, completion=completion, tardiness=tardiness)
            # 基于测试算例计算两个目标值
            tardiness_c, completion = self.run_one_epoch(environment=self.environment, policy_number=self.policy_completion)
            policy_objectives_dict[self.policy_completion]["completion"], policy_objectives_dict[self.policy_completion]["tardiness"] = completion, tardiness_c
            tardiness, completion_t = self.run_one_epoch(environment=self.environment, policy_number=self.policy_tardiness)
            policy_objectives_dict[self.policy_tardiness]["completion"], policy_objectives_dict[self.policy_tardiness]["tardiness"] = completion_t, tardiness
            for policy in self.policy_weight_tuple:
                tardiness_p, completion_p = self.run_one_epoch(environment=self.environment, policy_number=policy, completion=completion, tardiness=tardiness)
                policy_objectives_dict[policy]["completion"], policy_objectives_dict[policy]["tardiness"] = completion_p, tardiness_p
            completion_list = [policy_objectives_dict[p]["completion"] for p in self.policy_tuple]
            tardiness_list = [policy_objectives_dict[p]["tardiness"] for p in self.policy_tuple]
            # 训练曲线
            vis.line(X=[self.episode_number], Y=[sum(completion_list) / self.actor_number], win=window_1, update='append')
            vis.line(X=[self.episode_number], Y=[sum(tardiness_list) / self.actor_number], win=window_2, update='append')
            add_data_object.add_data([self.episode_number,
                                      sum(completion_list) / self.actor_number,
                                      sum(tardiness_list) / self.actor_number,
                                      min(completion_list), min(tardiness_list)])  # 保存数据
            self.episode_number += 1  # 更新周期数
            # 判断是否运行多策略进化机制
            if self.episode_number % 30 == 0:
                self.multi_policy_update(policy_objectives_dict)
            # 保存一次策略网络参数
            if self.completion_avg > sum(completion_list) / self.actor_number and self.tardiness_avg > sum(tardiness_list) / self.actor_number:
                self.completion_avg = sum(completion_list) / self.actor_number
                self.tardiness_avg = sum(tardiness_list) / self.actor_number
                self.save_policy_networks()

    def multi_policy_update(self, policy_objectives_dict):
        """多策略进化机制"""
        policy_weights_dict = {policy: [] for policy in self.policy_tuple}
        actor_network_dict = copy.deepcopy(self.actor_new_dict)
        critic_network_dict = copy.deepcopy(self.critic_dict)
        # 计算各策略网络在每个权重向量下的ge值+更新各策略网络参数
        for policy in self.policy_tuple:
            policy_weights_dict[policy] = tuple([self.weight_vector_dict[p][0]*(policy_objectives_dict[policy]["completion"]/self.completion_min) +
                                                 self.weight_vector_dict[p][1]*(policy_objectives_dict[policy]["tardiness"]/self.tardiness_min)
                                                 for p in self.policy_tuple])
            # 基于ge值更新各策略网络参数
            min_policy = policy_weights_dict[policy].index(min(policy_weights_dict[policy]))
            self.soft_update_of_target_network(actor_network_dict[min_policy], self.actor_new_dict[policy], self.hyper_parameters["tau"])
            self.soft_update_of_target_network(critic_network_dict[min_policy], self.critic_dict[policy], self.hyper_parameters["tau"])

    def save_policy_networks(self):
        """保存多个策略网络"""
        file_path = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/agents/MPPPO/policy_networks' + agent_version + '/'
        for policy in self.policy_tuple:
            actor_net = self.actor_new_dict[policy]
            torch.save(actor_net.state_dict(), file_path + 'policy_network' + str(policy) + '.path')

    def run_one_epoch(self, environment, policy_number, completion=None, tardiness=None):
        """用于跟踪训练效果"""
        # 提取当前策略网络
        self.actor_new = self.actor_new_dict[policy_number]  # 策略网络
        self.actor_old = self.actor_old_dict[policy_number]  # 目标策略网络
        self.weight_vector = self.weight_vector_dict[policy_number]  # 当前策略网络权重
        # 重置环境
        self.state = environment.reset()  # 重置环境到初始状态
        self.done = False
        while not self.done:
            self.action, self.log_prob = self.pick_action_and_log_prob(policy=self.actor_new, state=self.state, epsilon_exploration=0)
            self.next_state, self.reward, self.done = environment.step(self.action, weight_vector=self.weight_vector, completion=completion, tardiness=tardiness)
            self.state = self.next_state  # 更新当前状态

        return environment.delay_time_sum, environment.completion_time

    def run_one_policy_network(self, environment, policy_number, completion=None, tardiness=None):
        """特定策略网络在特定环境下运行一个周期"""
        # 提取当前策略网络
        self.actor_new = self.actor_new_dict[policy_number]  # 策略网络对象字典
        self.actor_old = self.actor_old_dict[policy_number]  # 目标策略网络对象字典
        self.actor_optimizer = self.actor_optimizer_dict[policy_number]  # 策略网络优化器对象字典
        self.critic_local = self.critic_dict[policy_number]  # 评论家网络
        self.critic_optimizer = self.critic_optimizer_dict[policy_number]  # 评论家网络优化器
        self.weight_vector = self.weight_vector_dict[policy_number]  # 当前策略网络权重
        # 随机探索
        exploration_epsilon = 1 / (1.0 + (self.episode_number / self.epsilon_decay_denominator))
        exploration = max(0.0, random.uniform(exploration_epsilon / 3.0, exploration_epsilon * 3.0))
        # 重置环境
        self.actor_old_log_prob = []  # 旧网络的动作概率
        self.memory.clear()  # 清空缓存数据
        self.state = environment.reset()  # 重置环境到初始状态
        self.done = False
        while not self.done:
            self.action, self.log_prob = self.pick_action_and_log_prob(policy=self.actor_new, state=self.state, epsilon_exploration=exploration)
            self.actor_old_log_prob.append(self.log_prob)
            self.next_state, self.reward, self.done = environment.step(self.action, weight_vector=self.weight_vector, completion=completion, tardiness=tardiness)
            self.save_experience(memory=self.memory)  # 保存经验条
            self.state = self.next_state  # 更新当前状态
        # 采样缓存中的经验进行学习
        self.episode_states, self.episode_actions, self.episode_rewards, self.episode_next_states, self.episode_dones \
            = self.sample_experiences(memory=self.memory)  # 采样批量数据
        self.actor_old_log_prob = torch.tensor(self.actor_old_log_prob).float().detach().to(self.device).detach()
        self.discounted_returns = self.calculate_discounted_returns()
        if self.hyper_parameters["normalized_rewards"]:  # 回报归一化
            self.discounted_returns = (self.discounted_returns - self.discounted_returns.min())/(self.discounted_returns.max() - self.discounted_returns.min() + 1e-8)
        if self.hyper_parameters["standardized_rewards"]:  # 回报标准化
            self.discounted_returns = (self.discounted_returns - self.discounted_returns.mean())/(self.discounted_returns.std() + 1e-8)
        self.critic_targets = self.discounted_returns
        self.advantages = self.discounted_returns - self.critic_local(self.episode_states).squeeze(1).detach()
        self.critic_actor_learn()  # 训练评论家+策略网络
        # 一轮学习完毕
        self.memory.clear()  # 清空缓存数据
        self.actor_old_log_prob = []  # 旧网络的动作概率
        self.equalise_policies()  # 更新旧策略网络参数

        return environment.delay_time_sum, environment.completion_time

    def pick_action_and_log_prob(self, policy, state, epsilon_exploration=None):
        """基于策略采样一个动作"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        actor_output = policy.forward(state)
        action_distribution = create_actor_distribution(self.action_types, actor_output, self.action_size)
        action = action_distribution.sample().cpu().numpy()
        action = int(action)
        if random.random() <= epsilon_exploration:
            action = random.randint(0, self.action_size - 1)
        else:
            action = action
        action_log_prob = self.calculate_log_action_probability(action, action_distribution)
        return action, action_log_prob

    def calculate_log_action_probability(self, actions, action_distribution):
        """计算所选动作的log概率"""
        policy_distribution_log_prob = action_distribution.log_prob(torch.Tensor([actions]).to(self.device))
        return policy_distribution_log_prob

    def sample_experiences(self, memory=None):
        """采样一个批次的经验条"""
        return memory.sample()

    def get_critic_value(self, policy, state):
        """返回评论家网络值"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # 状态转为tensor类型
        critic_output = policy.forward(state)
        return critic_output

    def calculate_discounted_returns(self):
        """
        计算一集的累计折现收益，然后我们将在学习迭代中使用：蒙特卡洛估计计算V(s)_target 值
        """
        discounted_returns = [0]
        for ix in range(len(self.episode_states)):
            return_value = self.episode_rewards[-(ix + 1)] + self.discount_rate*discounted_returns[-1]
            discounted_returns.append(return_value)
        discounted_returns = discounted_returns[1:]
        discounted_returns = discounted_returns[::-1]
        discounted_returns = torch.tensor(discounted_returns).to(self.device).detach()
        return discounted_returns

    def critic_actor_learn(self):
        """策略网络学习"""
        for _ in range(self.learning_iterations_per_round_critic):
            critic_outs = self.critic_local(self.episode_states).squeeze(1)
            critic_loss = F.mse_loss(critic_outs, self.critic_targets)
            critic_loss = critic_loss.clone().detach().requires_grad_(True)
            self.take_critic_optimisation_step(critic_loss)  # 评论家优化
            all_ratio_of_policy_probabilities = self.calculate_all_ratio_of_policy_probabilities(self.actor_old_log_prob)
            actor_loss = self.calculate_actor_loss([all_ratio_of_policy_probabilities], self.advantages)
            self.take_policy_new_optimisation_step(actor_loss)  # 策略网络优化

    def calculate_all_ratio_of_policy_probabilities(self, actor_old_log_prob):
        """对于每个操作，计算新策略选择该操作的概率与旧策略选择该操作的概率之比"""
        states = [state for state in self.episode_states]
        actions = [[action] for action in self.episode_actions]
        states = torch.stack([torch.as_tensor(state).float().to(self.device) for state in states])
        actions = torch.stack([torch.as_tensor(action).float().to(self.device) for action in actions])
        actions = actions.view(-1, len(states))
        new_policy_distribution_log_prob = self.calculate_log_probability_of_actions(self.actor_new, states, actions).squeeze(0)
        old_policy_distribution_log_prob = actor_old_log_prob
        ratio_of_policy_probabilities = torch.exp(new_policy_distribution_log_prob) / (torch.exp(old_policy_distribution_log_prob) + 1e-8)
        return ratio_of_policy_probabilities

    def calculate_log_probability_of_actions(self, policy, states, actions):
        """计算给定策略和状态下某动作的发生log概率"""
        policy_output = policy.forward(states).to(self.device)
        policy_distribution = create_actor_distribution(self.action_types, policy_output, self.action_size)
        policy_distribution_log_prob = policy_distribution.log_prob(actions)
        return policy_distribution_log_prob

    def calculate_actor_loss(self, all_ratio_of_policy_probabilities, advantages):
        """计算策略损失"""
        all_ratio_of_policy_probabilities = torch.squeeze(torch.stack(all_ratio_of_policy_probabilities))
        all_ratio_of_policy_probabilities = torch.clamp(input=all_ratio_of_policy_probabilities, min=-sys.maxsize, max=sys.maxsize)
        potential_loss_value_1 = advantages * all_ratio_of_policy_probabilities
        potential_loss_value_2 = advantages * self.clamp_probability_ratio(all_ratio_of_policy_probabilities)
        actor_loss = torch.min(potential_loss_value_1, potential_loss_value_2)
        actor_loss = -torch.mean(actor_loss)
        return actor_loss

    def clamp_probability_ratio(self, value):
        """在由超参数clip epsilon确定的一定范围内裁剪出一个值"""
        return torch.clamp(input=value, min=1.0 - self.hyper_parameters["clip_epsilon"], max=1.0 + self.hyper_parameters["clip_epsilon"])

    def take_policy_new_optimisation_step(self, actor_loss):
        """新策略优化"""
        self.actor_optimizer.zero_grad()  # reset gradients to 0
        actor_loss.backward()  # this calculates the gradients
        torch.nn.utils.clip_grad_norm_(self.actor_new.parameters(), self.hyper_parameters["gradient_clipping_norm"])  # 梯度裁剪
        self.actor_optimizer.step()  # 优化参数

    def take_critic_optimisation_step(self, critic_loss):
        """评论家优化"""
        self.critic_optimizer.zero_grad()  # 重置梯度为0
        critic_loss.backward()  # 计算梯度
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.hyper_parameters["gradient_clipping_norm"])  # 梯度裁剪
        self.critic_optimizer.step()  # 优化参数

    def equalise_policies(self):
        """更新旧策略网络参数"""
        for old_param, new_param in zip(self.actor_old.parameters(), self.actor_new.parameters()):
            old_param.algorithm_means.copy_(new_param.algorithm_means)

    def time_for_critic_and_actor_to_learn(self):
        """返回布尔值，指示是否有足够的经验可供演员和评论家学习"""
        return self.enough_experiences_to_learn_from(memory=self.memory, batch_size=self.batch_size)


# 测试算法
if __name__ == '__main__':
    ppo_object = PPO()
    ppo_object.run_n_episodes()
