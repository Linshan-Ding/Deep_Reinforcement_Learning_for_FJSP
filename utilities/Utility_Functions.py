import math

import numpy as np
from abc import ABCMeta
import torch
from nn_builder.pytorch.NN import NN
from torch.distributions import Categorical, normal, MultivariateNormal


def abstract(cls):
    return ABCMeta(cls.__name__, cls.__bases__, dict(cls.__dict__))


def save_score_results(file_path, results):
    """在给定的路径保存结果"""
    np.save(file_path, results)


def standardized_rewards(rewards):
    """标准化回报，均值为0，标准差为1"""
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return (rewards - mean_reward) / (std_reward + 1e-8)  # 1e-8 added for stability


def normalized_rewards(rewards):
    """归一化回报值"""
    max_value = np.max(rewards)
    min_value = np.min(rewards)
    rewards -= min_value
    rewards = rewards.astype(float)
    rewards /= (max_value - min_value + 1e-5)
    return rewards


def create_actor_distribution(action_types, actor_output, action_size):
    """生成一个分布，并根据分布采用一个动作"""
    if action_types == "DISCRETE":
        assert actor_output.size()[1] == action_size, "Actor output the wrong size"
        action_distribution = Categorical(actor_output)  # this creates a distribution to sample from
    else:
        assert actor_output.size()[1] == action_size * 2, "Actor output the wrong size"
        means = actor_output[:, :action_size].squeeze(0)
        stds = actor_output[:,  action_size:].squeeze(0)
        if len(means.shape) == 2:
            means = means.squeeze(-1)
        if len(stds.shape) == 2:
            stds = stds.squeeze(-1)
        if len(stds.shape) > 1 or len(means.shape) > 1:
            raise ValueError("Wrong mean and std shapes - {} -- {}".format(stds.shape, means.shape))
        action_distribution = normal.Normal(means.squeeze(0), torch.abs(stds))
    return action_distribution


class SharedAdam(torch.optim.Adam):
    """
    创建可在进程之间共享的adam优化器对象。对于像A3C这样的算法很有用.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                amsgrad = group['amsgrad']
                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                # 衰减一、二矩运行平均系数
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


def flatten_action_id_to_actions(action_id_to_actions, global_action_id_to_primitive_action, num_primitive_actions):
    """将action_id_to_actions字典中的值转换回它们所表示的基本动作"""
    flattened_action_id_to_actions = {}
    for key in action_id_to_actions.keys():
        actions = action_id_to_actions[key]
        raw_actions = backtrack_action_to_primitive_actions(actions, global_action_id_to_primitive_action, num_primitive_actions)
        flattened_action_id_to_actions[key] = raw_actions
    return flattened_action_id_to_actions


def backtrack_action_to_primitive_actions(action_tuple, global_action_id_to_primitive_action, num_primitive_actions):
    """Converts an action tuple back to the primitive actions it represents in a recursive way."""
    print("Recursing to backtrack on ", action_tuple)
    primitive_actions = range(num_primitive_actions)
    if all(action in primitive_actions for action in action_tuple): return action_tuple #base case
    new_action_tuple = []
    for action in action_tuple:
        if action in primitive_actions: new_action_tuple.append(action)
        else:
            converted_action = global_action_id_to_primitive_action[action]
            print(new_action_tuple)
            new_action_tuple.extend(converted_action)
            print("Should have changed: ", new_action_tuple)
    new_action_tuple = tuple(new_action_tuple)
    return backtrack_action_to_primitive_actions(new_action_tuple)
