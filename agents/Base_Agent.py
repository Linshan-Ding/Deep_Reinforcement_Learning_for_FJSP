"""基础智能体类"""
import random
import numpy as np
import torch
import time
from torch.optim import optimizer


class Base_Agent():
    def __init__(self):
        self.action_size = None  # 动作维度
        self.state_size = None  # 状态维度
        self.episode_number = 0  # 周期数
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备选择
        print("选择的设备：", self.device)
        self.turn_off_exploration = False  # 探索控制
        # 通用环境相关参数和周期参数定义
        self.state = None
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False

        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []

    def step(self):
        """
        在对应智能体中重写step方法
        """
        raise ValueError("step方法需要在对应智能体中重写")

    def get_state_size(self):
        """
        环境状态维度-用于构建神经网络
        """
        return ValueError("该方法需要在对应智能体中重写")

    def reset_game(self):
        """重置环境"""
        self.state = None
        self.next_state = None
        self.action = None
        self.reward = None
        self.done = False
        self.episode_states = []
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_next_states = []
        self.episode_dones = []

    def track_episodes_data(self):
        """保存最近几集的数据"""
        self.episode_states.append(self.state)
        self.episode_actions.append(self.action)
        self.episode_rewards.append(self.reward)
        self.episode_next_states.append(self.next_state)
        self.episode_dones.append(self.done)

    def enough_experiences_to_learn_from(self, memory, batch_size):
        """布尔值表示内存缓冲区中是否有足够的经验可供学习"""
        return len(memory) > batch_size

    def save_experience(self, memory=None, experience=None):
        """将最近的经验保存到内存缓冲区"""
        if experience is None:
            experience = self.state, self.action, self.reward, self.next_state, self.done
        memory.add_experience(*experience)

    def take_optimisation_step(self, optimizer, network, loss, clipping_norm=None, retain_graph=False):
        """通过计算给定损失的梯度然后更新参数来进行优化步骤"""
        if not isinstance(network, list):
            network = [network]
        optimizer.zero_grad()  # 重置梯度为 0
        loss.backward(retain_graph=retain_graph)  # 计算梯度
        if clipping_norm is not None:
            for net in network:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clipping_norm)  # 裁剪梯度以帮助稳定训练
        optimizer.step()  # 更新梯度

    def soft_update_of_target_network(self, local_model, target_model, tau=0.005):
        """软更新目标网络"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def turn_on_any_epsilon_greedy_exploration(self):
        """开启所有贪心勘探策略的勘探"""
        print("Turning on epsilon greedy exploration")
        self.turn_off_exploration = False

    def turn_off_any_epsilon_greedy_exploration(self):
        """关闭所有贪心勘探策略下的勘探"""
        print("Turning off epsilon greedy exploration")
        self.turn_off_exploration = True

    def freeze_all_but_output_layers(self, network):
        """冻结网络除输出层外的所有层"""
        print("Freezing hidden layers")
        for param in network.named_parameters():
            param_name = param[0]
            assert "hidden" in param_name or "output" in param_name or "embedding" in param_name, \
                "Name {} of network layers not understood".format(param_name)
            if "output" not in param_name:
                param[1].requires_grad = False

    def unfreeze_all_layers(self, network):
        """解冻网络的所有层"""
        print("Unfreezing all layers")
        for param in network.parameters():
            param.requires_grad = True

    @staticmethod
    def move_gradients_one_model_to_another(from_model, to_model, set_from_gradients_to_zero=False):
        """从from_model复制梯度到to_model"""
        for from_model, to_model in zip(from_model.parameters(), to_model.parameters()):
            to_model._grad = from_model.grad.clone()
            if set_from_gradients_to_zero:
                from_model._grad = None

    @staticmethod
    def copy_model_over(from_model, to_model):
        """将模型参数从from_model复制到to_model"""
        to_model.load_state_dict(from_model.state_dict())

    @staticmethod
    def copy_model_over_dict(from_model, to_model):
        """将模型参数从from_model复制到to_model"""
        # 复制模型参数
        to_model_state_dict = to_model.state_dict()
        from_model_state_dict = from_model.state_dict()
        # 更新to_model的参数为from_model的参数
        to_model_state_dict.update(from_model_state_dict)
        # 加载更新后的参数到to_model
        to_model.load_state_dict(to_model_state_dict)

