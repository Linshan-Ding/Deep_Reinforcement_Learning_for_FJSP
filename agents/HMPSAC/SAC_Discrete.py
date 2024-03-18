"""
上层控制策略网络训练算法
"""
import random
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch import nn
import numpy as np
from agents.Base_Agent import Base_Agent
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from utilities.Utility_Functions import create_actor_distribution
from utilities.data_structures.Config import Config
from utilities.OU_Noise import OU_Noise
from environments.MO_DFJSP import MO_DFJSP_Environment
from visdom import Visdom
from utilities.Utility_Class import AddData

# 训练结果数据保存位置
agent_version = '_v3.1'
path_file_name = 'D:/Python project/Deep_Reinforcement_learning_FJSP/results/HMPSAC/training' + agent_version + '.csv'
add_data_object = AddData(path_file_name)
add_data_object.add_data(['epoch', 'makespan', 'tardiness', 'energy'])
# 监控训练过程
vis = Visdom()
window_1 = 'completion_time' + agent_version
title_1 = window_1
vis.line(X=[0], Y=[0], win=window_1, opts=dict(title=title_1, xlabel='epoch', ylable='completion_time', font=dict(family='Times New Roman')))
window_2 = 'total_delay_time' + agent_version
title_2 = window_2
vis.line(X=[0], Y=[0], win=window_2, opts=dict(title=title_2, xlabel='epoch', ylable='total_delay_time', font=dict(family='Times New Roman')))
window_3 = 'total_energy_consumption' + agent_version
title_3 = window_3
vis.line(X=[0], Y=[0], win=window_3, opts=dict(title=title_3, xlabel='epoch', ylable='total_energy_consumption', font=dict(family='Times New Roman')))

# 全局参数定义
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
TRAINING_EPISODES_PER_EVAL_EPISODE = 10
EPSILON = 1e-6


class TaskPolicyNet(nn.Module):
    def __init__(self, input_size_1, hidden_size, hidden_layer_1, output_size_1):
        super(TaskPolicyNet, self).__init__()
        self.name = "task_policy"
        # 定义工序策略网络输入层
        self.layers_1 = nn.ModuleList([nn.Linear(input_size_1, hidden_size), nn.ReLU()])
        # 定义工序策略网络隐藏层
        for i in range(hidden_layer_1 - 1):
            self.layers_1.append(nn.Linear(hidden_size, hidden_size))
            self.layers_1.append(nn.ReLU())
        # 定义工序策略网络输出层
        self.layers_1.append(nn.Linear(hidden_size, output_size_1))

    def forward(self, x):
        for layer in self.layers_1:
            x = layer(x)
        x = F.softmax(x, dim=-1)
        return x


class MachinePolicyNet(nn.Module):
    def __init__(self, input_size_2, hidden_size, hidden_layer_2, output_size_2):
        super(MachinePolicyNet, self).__init__()
        self.name = "machine_policy"
        # 定义机器策略网络输入层
        self.layers_2 = nn.ModuleList([nn.Linear(input_size_2, hidden_size), nn.ReLU()])
        # 定义机器策略网络隐藏层
        for i in range(hidden_layer_2 - 1):
            self.layers_2.append(nn.Linear(hidden_size, hidden_size))
            self.layers_2.append(nn.ReLU())
        # 定义机器策略网络输出层
        self.layers_2.append(nn.Linear(hidden_size, output_size_2))

    def forward(self, x):
        for layer in self.layers_2:
            x = layer(x)
        x = F.softmax(x, dim=-1)
        return x


class PolicyNet(nn.Module):
    """工序策略网络"""
    def __init__(self, input_size, hidden_size, hidden_layer, output_size):
        super(PolicyNet, self).__init__()
        self.name = "task_policy"
        # 定义工序策略网络输入层
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size), nn.ReLU()])
        # 定义工序策略网络隐藏层
        for i in range(hidden_layer - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())
        # 定义工序策略网络输出层
        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = F.softmax(x, dim=-1)
        return x


class CriticNet(nn.Module):
    """评论家网络"""
    def __init__(self, input_size, hidden_size, hidden_layer, output_size, seed=None):
        super(CriticNet, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
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


class SAC_Discrete(Base_Agent, Config):
    """离散动作软演员评论家算法"""
    def __init__(self):
        Base_Agent.__init__(self)  # 继承基础智能体类
        Config.__init__(self)  # 继承算法超参数类
        self.config = Config()  # 算法控制参数
        self.agent = "HMP_SAC"
        self.hyper_parameters = self.hyper_parameters[self.agent]  # 算法控制参数
        self.action_types = "DISCRETE"
        assert self.action_types == "DISCRETE", "Action types must be discrete. Use SAC instead for continuous actions"
        # 定义测试环境
        self.path = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/data/HMPSAC'  # 测试算例的存储位置
        self.file_name = 'DDT0.5_M10_S1'  # 测试算例的文件夹名字
        self.environment = MO_DFJSP_Environment(use_instance=False, path=self.path, file_name=self.file_name)  # 测试环境
        # 超参数
        self.num_episodes_to_run = self.hyper_parameters["num_episodes_to_run"]  # 训练周期数
        self.learning_rate = self.hyper_parameters["learning_rate"]  # 学习率
        self.discount_rate = self.hyper_parameters["discount_rate"]  # 折扣率
        self.buffer_size = self.hyper_parameters["buffer_size"]  # 回放记忆缓存大小
        self.batch_size = self.hyper_parameters["batch_size"]  # 采样批量
        self.gradient_clipping_norm = self.hyper_parameters["gradient_clipping_norm"]  # 梯度裁剪
        # 算法运行参数
        self.global_step_number = 0  # 运行总步数
        self.episode_number = 0  # 更新周期数
        # 环境状态维度参数
        self.state_size = 30
        self.action_size = 3
        self.actor_input_size = self.state_size
        self.critic_input_size = self.state_size
        # 定义评论家网络
        self.critic_local = CriticNet(input_size=self.critic_input_size, hidden_size=200, hidden_layer=3,
                                      output_size=self.action_size, seed=random.randint(1, 10)).to(self.device)
        self.critic_local_2 = CriticNet(input_size=self.critic_input_size, hidden_size=200, hidden_layer=3,
                                        output_size=self.action_size, seed=random.randint(11, 20)).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=self.hyper_parameters["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(), lr=self.hyper_parameters["learning_rate"], eps=1e-4)
        self.critic_target = CriticNet(input_size=self.critic_input_size, hidden_size=200, hidden_layer=3, output_size=self.action_size).to(self.device)
        self.critic_target_2 = CriticNet(input_size=self.critic_input_size, hidden_size=200, hidden_layer=3, output_size=self.action_size).to(self.device)
        self.memory = Replay_Buffer(self.hyper_parameters["buffer_size"], self.hyper_parameters["batch_size"], device=self.device)
        # 定义策略网络
        self.actor_local = PolicyNet(input_size=self.actor_input_size, hidden_size=200, hidden_layer=3, output_size=self.action_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=self.hyper_parameters["learning_rate"], eps=1e-4)
        # 定义熵损失
        self.automatic_entropy_tuning = self.hyper_parameters["automatically_tune_entropy_hyper_parameter"]
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyper_parameters["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyper_parameters["entropy_term_weight"]
        assert not self.hyper_parameters["add_extra_noise"], "目前没有为SAC的离散版本添加额外的噪声选项"
        self.add_extra_noise = False
        self.do_evaluation_iterations = self.hyper_parameters["do_evaluation_iterations"]
        # 导入三个训练好的目标策略网络
        self.objectives_policy = {'makespan': 0, 'tardiness': 1, 'energy': 2}
        self.action_size_dict = {'task': 12, 'machine': 10}
        self.policy_dict = {0: {}, 1: {}, 2: {}}
        self.load_policy_model()  # 加载网络

    def load_policy_model(self):
        """加载三目标策略网络"""
        for objective, policy in self.objectives_policy.items():
            file_path = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/results/HMPSAC/policy_networks_v5.' + str(policy + 1) + '/'
            actor_net_task = TaskPolicyNet(input_size_1=30, hidden_size=200, hidden_layer_1=3, output_size_1=12).to(self.device)
            actor_net_task.load_state_dict(torch.load(file_path + 'actor_task_model.path'))
            self.policy_dict[policy]['task'] = actor_net_task
            actor_net_machine = MachinePolicyNet(input_size_2=31, hidden_size=200, hidden_layer_2=3, output_size_2=10).to(self.device)
            actor_net_machine.load_state_dict(torch.load(file_path + 'actor_machine_model.path'))
            self.policy_dict[policy]['machine'] = actor_net_machine

    def run_n_episodes(self):
        """运行N个周期"""
        for epoch in range(self.num_episodes_to_run):
            environment = self.environment
            objectives_values = []
            # 基于三个目标策略网络获得实例的三个目标基准值
            for objective, policy in self.objectives_policy.items():
                # 基于控制策略网络收集训练数据：选择目标+选择工序规则+选择机器规则+基于复合动作step环境
                self.state = environment.reset()
                self.done = environment.done
                while not self.done:
                    action_task = self.pick_lower_action(policy=self.policy_dict[policy]['task'], state=self.state,
                                                         action_size=self.action_size_dict['task'])
                    state_machine = np.append(self.state, action_task)  # 带选择的工序规则信息的状态
                    action_machine = self.pick_lower_action(policy=self.policy_dict[policy]['machine'], state=state_machine,
                                                            action_size=self.action_size_dict['machine'])
                    action_task_machine = np.array([action_task, action_machine])
                    self.next_state, self.reward, self.done = environment.step(action_task_machine, reward_policy=0, completion=None,
                                                                               tardiness=None, energy_consumption=None)
                    self.state = self.next_state
                # 存入对应的目标值
                objectives_values.append([environment.completion_time, environment.delay_time_sum, environment.energy_consumption])
            # 提取最小目标值
            objectives_arr = np.array(objectives_values)
            objectives_value = np.min(objectives_arr, axis=0)
            # 基于控制策略网络收集训练数据：选择目标+选择工序规则+选择机器规则+基于复合动作step环境
            self.state = environment.reset()
            self.done = environment.done
            while not self.done:
                self.action = self.pick_action(self.state)  # 选择目标网络
                action_task = self.pick_lower_action(policy=self.policy_dict[self.action]['task'], state=self.state,
                                                     action_size=self.action_size_dict['task'])
                state_machine = np.append(self.state, action_task)  # 带选择的工序规则信息的状态
                action_machine = self.pick_lower_action(policy=self.policy_dict[self.action]['machine'], state=state_machine,
                                                        action_size=self.action_size_dict['machine'])
                action_task_machine = np.array([action_task, action_machine])
                self.next_state, self.reward, self.done = environment.step(action_task_machine, reward_policy=3, completion=objectives_value[0],
                                                                           tardiness=objectives_value[1], energy_consumption=objectives_value[2])
                self.save_experience(memory=self.memory, experience=(self.state, self.action, self.reward, self.next_state, self.done))
                if self.time_for_critic_and_actor_to_learn():
                    for _ in range(self.hyper_parameters["learning_updates_per_learning_session"]):
                        self.learn()
                self.state = self.next_state
                self.global_step_number += 1

            print("总回报：", environment.reward_sum)
            vis.line(X=[self.episode_number], Y=[environment.completion_time], win=window_1, update='append')
            vis.line(X=[self.episode_number], Y=[environment.delay_time_sum], win=window_2, update='append')
            vis.line(X=[self.episode_number], Y=[environment.energy_consumption], win=window_3, update='append')
            self.episode_number += 1

    def pick_action(self, state):
        """采样动作"""
        if self.global_step_number < self.hyper_parameters["min_steps_before_learning"]:
            action = random.randint(0, self.action_size - 1)
        else:
            action = self.actor_pick_action(state=state)
        return action

    def actor_pick_action(self, state):
        """采样一个动作"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        action, _, _ = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def produce_action_and_action_info(self, state):
        """输入状态，采样动作，各动作的概率和log概率，最大概率"""
        action_probabilities = self.actor_local(state)
        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = create_actor_distribution(self.action_types, action_probabilities, self.action_size)
        action = action_distribution.sample().cpu()  # 动作采样
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

    def pick_lower_action(self, policy, state, action_size):
        """基于策略采样一个动作"""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        actor_output = policy.forward(state)
        action_distribution = create_actor_distribution(self.action_types, actor_output, action_size)
        action = action_distribution.sample().cpu().numpy()
        action = int(action)
        return action

    def time_for_critic_and_actor_to_learn(self):
        """判断是否具有充足的经验去学习"""
        return self.global_step_number > self.hyper_parameters["min_steps_before_learning"] and \
               self.enough_experiences_to_learn_from(self.memory, self.batch_size) and \
               self.global_step_number % self.hyper_parameters["update_every_n_steps"] == 0

    def learn(self):
        """策略网络+评论家网络+最大熵：参数更新"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch, mask_batch)
        self.update_critic_parameters(qf1_loss, qf2_loss)
        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        if self.automatic_entropy_tuning:
            alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        else:
            alpha_loss = None
        self.update_actor_parameters(policy_loss, alpha_loss)

    def sample_experiences(self):
        return self.memory.sample()

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        """计算两位评论家的损失。这是普通的q学习损失，除了额外的熵项被考虑在内"""
        with torch.no_grad():
            next_state_action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - done_batch) * self.hyper_parameters["discount_rate"] * min_qf_next_target

        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """计算策略网络损失"""
        action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)  # 计算熵损失
        return policy_loss, log_action_probabilities

    def calculate_entropy_tuning_loss(self, log_pi):
        """计算熵参数损失"""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_critic_parameters(self, critic_loss_1, critic_loss_2):
        """更新评论家参数"""
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1, self.hyper_parameters["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2, self.hyper_parameters["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target, self.hyper_parameters["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2, self.hyper_parameters["tau"])

    def update_actor_parameters(self, actor_loss, alpha_loss):
        """更新策略网络+熵参数"""
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss, self.hyper_parameters["gradient_clipping_norm"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()


# 测试算法
if __name__ == '__main__':
    sac_object = SAC_Discrete()
    sac_object.run_n_episodes()

