"""
多目标静态柔性作业车间调度环境+连续动作
目标：最大完工时间+总延期时间
"""
import copy
import random, math, time
import numpy as np
from environments.SO_FJSSP import SO_FJSSP_Environment
from utilities.Utility_Class import FigGan, MyError


class MO_FJSSP_Environment(SO_FJSSP_Environment):
    """多目标柔性作业车间静态调度环境"""
    def __init__(self, use_instance=True, **kwargs):
        super().__init__(use_instance=use_instance, **kwargs)
        self.state_size = 24  # 状态空间
        self.action_types = "CONTINUOUS"  # 动作类型
        self.observation_space = 9  # 观察的动态状态向量空间
        self.static_state_space = 6  # 静态状态向量空间
        self.order_object = self.order_dict[0]  # 实例订单对象
        self.static_state = self.static_state_extract()  # 静态特征
        self.reward_sum = 0  # 累计回报
        self.completion_time = 0  # 当前时间点完工时间
        self.completion_time_last = 0  # 上一时间步完工时间

    def reset(self):
        """重置环境状态"""
        self.reset_parameter()  # 初始化参数对象中的列表和字典
        self.reset_object_add(self.order_object)  # 新订单到达后更新各字典对象
        # 重置回报相关属性
        self.delay_time_sum_last = 0  # 上一时间步工件实际总延期时间
        self.delay_time_sum = 0  # 当前时间步工件实际总延期时间
        self.delay_time_sum_processed = 0  # 已完工工件的总延期时间
        self.delay_time_sum_unprocessed = 0  # 未完工工件的总延期时间
        self.delay_time_sum_unprocessed_last = 0
        self.reward_sum = 0  # 累计回报
        self.gap_ave_value_last = 0  # 上一时间步的工序类型gap均值
        self.completion_time_last = 0  # 上一时间步完工时间
        self.completion_time = 0
        # 重置时间步和当前时间点
        self.step_count = 0
        self.step_time = 0
        # 重置上一时间步和当前时间步观察到的状态
        self.last_observation_state = self.state_extract()  # 上一步观察到的状态 v(t-1)
        self.observation_state = self.state_extract()  # 当前时间步的状态 v(t)
        self.state_gap = self.observation_state - self.last_observation_state  # v(t) - v(t-1)
        self.state = np.concatenate((self.static_state, self.observation_state, self.state_gap), axis=0)  # 状态向量 [v(t), v(t) - v(t-1)]
        self.next_state = None  # 下一步状态
        self.reward = None  # 即时奖励
        self.done = False  # 是否为终止状态
        return self.state

    def static_state_extract(self):
        """提取任务相关静态状态"""
        M = self.machine_count  # 机器数
        R = self.kind_count  # 工件类型数
        N_ave = sum(self.order_object.count_kind[r] for r in self.kind_tuple)/self.kind_count  # 各工件类型的工件数均值
        N_std = math.sqrt(sum(math.pow(self.order_object.count_kind[r] - N_ave, 2)
                              for r in self.kind_tuple)/self.kind_count)  # 各工件类型的工件数标准差
        J_ave = sum(len(self.task_r_dict[r]) for r in self.kind_tuple)/self.kind_count  # 工件类型的工序数均值
        J_std = math.sqrt(sum(math.pow(len(self.task_r_dict[r]) - J_ave, 2)
                              for r in self.kind_tuple)/self.kind_count)
        return np.array([M, R, N_ave, N_std, J_ave, J_std])

    def state_extract(self):
        """
        提取状态向量
        更新相关参数
        """
        ct_m_ave = self.ct_m_ave  # 机器的平均完工时间
        ct_m_std = math.sqrt(sum(math.pow(machine_object.time_end - ct_m_ave, 2) for m, machine_object
                                 in self.machine_dict.items())/self.machine_count)   # 2-机器完工时间标准差
        cro_ave = sum(kind_task_object.finish_rate for (r, j), kind_task_object in
                      self.kind_task_dict.items())/len(self.kind_task_tuple)  # 3-工序类型完工率均值
        cro_std = math.sqrt(sum(math.pow(kind_task_object.finish_rate - cro_ave, 2) for (r, j), kind_task_object
                                in self.kind_task_dict.items())/len(self.kind_task_tuple))  # 4-工序类型完工率标准差
        gap_ave = sum(kind_task_object.gap_rate for (r, j), kind_task_object
                      in self.kind_task_dict.items())/len(self.kind_task_tuple)  # 5-工序类型gap_rj均值
        gap_std = math.sqrt(sum(math.pow(kind_task_object.gap_rate - gap_ave, 2) for (r, j), kind_task_object
                                in self.kind_task_dict.items())/len(self.kind_task_tuple))  # 6-工序类型gap_rj标准差
        # 返回7-工序实际和8-估计延迟率、9-工件实际和10-估计延迟率+更新相关参数
        dro_a, dro_e, drj_a, drj_e = self.update_parameter()
        return np.array([ct_m_std, cro_ave, cro_std, gap_ave, gap_std, dro_a, dro_e, drj_a, drj_e])

    def step(self, action, weight_vector=None, completion=None, tardiness=None):
        """根据动作选择工序选择规则+机器分配规则"""
        self.gap_ave_value_last = self.gap_ave_value
        rj_selected = self.task_assignment(action)  # 选择的工序类型
        m_selected = self.machine_assignment(rj_selected)  # 选择的机器
        # 定义相关对象
        job_object_selected = self.kind_task_dict[rj_selected].job_now_list[0]  # 工件对象
        task_object_selected = job_object_selected.task_unprocessed_list[0]  # 工序对象
        kind_task_object_selected = self.kind_task_dict[rj_selected]  # 工序类型对象
        kind_object_selected = self.kind_dict[rj_selected[0]]  # 工件类型对象
        machine_object_selected = self.machine_dict[m_selected]  # 机器对象
        # 更新工序对象属性
        task_object_selected.time_begin = self.step_time  # 工序开工时间
        task_object_selected.machine = m_selected  # 选择的机器
        task_object_selected.time_end = self.step_time + self.time_mrj_dict[m_selected][rj_selected]  # 完工时间
        # 更新工件对象属性
        job_object_selected.task_list.append(task_object_selected)  # 更新已分配机器工序对象列表
        job_object_selected.task_unprocessed_list.remove(task_object_selected)  # 更新未分配机器工序对象列表
        # 更新工序类型对象
        kind_task_object_selected.job_now_list.remove(job_object_selected)  # 更新处于该工序段的工件对象列表
        kind_task_object_selected.job_unprocessed_list.remove(job_object_selected)  # 更新该工序段未加工的工件对象列表
        kind_task_object_selected.task_unprocessed_list.remove(task_object_selected)  # 更新该工序段未加工的工序对象列表
        kind_task_object_selected.task_processed_list.append(task_object_selected)  # 更新该工序段已分配机器的工序对象列表
        # 更新机器对象
        machine_object_selected.state = 1  # 更新机器状态
        machine_object_selected.time_end = task_object_selected.time_end  # 更新机器完工时间
        machine_object_selected.task_list.append(task_object_selected)  # 更新机器已加工工序对象列表
        machine_object_selected.job_object = job_object_selected  # 更新机器正在加工的工件对象
        machine_object_selected.unprocessed_rj_dict[rj_selected] -= 1  # 更新该机器未加工工序rj的数量
        self.completion_time = max(self.completion_time, task_object_selected.time_end)
        # 更新工件类型对象和当前步实际总延期时间
        if len(job_object_selected.task_unprocessed_list) == 0:
            kind_object_selected.job_unprocessed_list.remove(job_object_selected)
            self.delay_time_sum_processed += max(task_object_selected.time_end - task_object_selected.due_date, 0)
        # 判断是否移动时钟
        while len(self.kind_task_available_list) == 0:
            time_point_list = [self.machine_dict[m].time_end for m in self.machine_tuple
                               if self.machine_dict[m].time_end > self.step_time]
            self.step_time = min(time_point_list)
            # 更新对象相关属性: 工序类型阶段的工件对象列表
            for m, machine_object in self.machine_dict.items():
                if machine_object.time_end == self.step_time:
                    job_object = machine_object.job_object  # 刚加工完的工件对象
                    if len(job_object.task_unprocessed_list) > 0:
                        task_object = job_object.task_unprocessed_list[0]  # 新到达的工序对象
                        kind_task_object = self.kind_task_dict[(task_object.kind, task_object.task)]  # 对应的工序类型
                        kind_task_object.job_now_list.append(job_object)
                        sorted(kind_task_object.job_now_list, key=lambda x: x.number)
            # 更新机器状态和正在加工的工件对象
            for m, machine_object in self.machine_dict.items():
                if machine_object.time_end <= self.step_time:
                    machine_object.state = 0  # 更新机器状态
            # 更新流体相关属性：工序类型流体量、机器-工序类型流体量
            gap_time = self.step_time - self.order_arrive_time  # 流动时间
            for (r, j), kind_task_object in self.kind_task_dict.items():
                kind_task_object.fluid_unprocessed_number = kind_task_object.fluid_unprocessed_number_start - \
                                                            kind_task_object.fluid_rate_sum*gap_time
            for m, machine_object in self.machine_dict.items():
                for (r, j) in machine_object.kind_task_tuple:
                    machine_object.fluid_unprocessed_rj_dict[(r, j)] = \
                        machine_object.fluid_unprocessed_rj_arrival_dict[(r, j)] - \
                        gap_time*machine_object.fluid_process_rate_rj_dict[(r, j)]
            # 判断是否终止
            if sum(len(kind_object.job_unprocessed_list) for r, kind_object in self.kind_dict.items()) == 0:
                self.done = True
                break
        # 提取新的状态、计算回报值、判断周期循环是否结束
        self.step_count += 1
        self.last_observation_state = self.observation_state  # 上一步观察到的状态 v(t-1)
        self.delay_time_sum_unprocessed_last = self.delay_time_sum_unprocessed  # 更新上一时间步未完工工件的实际总延期时间
        # 初始化初始时间步
        self.observation_state = self.state_extract()  # 当前时间步的状态 v(t)
        self.state_gap = self.observation_state - self.last_observation_state
        self.next_state = np.concatenate((self.static_state, self.observation_state, self.state_gap), axis=0)
        self.delay_time_sum = self.delay_time_sum_processed + self.delay_time_sum_unprocessed  # 实际总延期时间
        self.reward = self.compute_reward(weight_vector, completion, tardiness)  # 即时奖励
        # print("即时回报值：", self.reward)
        self.reward_sum += self.reward  # 更新累计回报
        self.delay_time_sum_last = self.delay_time_sum  # 更新上一时间步实际总的延期时间
        self.completion_time_last = self.completion_time  # 更新上一步完工时间
        self.state = self.next_state
        return self.state, self.reward, self.done

    def task_assignment(self, action):
        """基于连续动作的工序选择规则"""
        # 按照gap_rj值进行从小到大排序
        sorted_gap = sorted(self.kind_task_available_list, key=lambda x: self.kind_task_dict[x].gap)
        pri_gap_arr = np.array([sorted_gap.index(x) + 1 for x in self.kind_task_available_list])  # gap值从大到小排序的次序
        # 按照urg_rj值进行从小到达排序
        sorted_urg = sorted(self.kind_task_available_list, key=lambda x: self.kind_task_delivery_urgency[x])
        pri_urg_arr = np.array([sorted_urg.index(x) + 1 for x in self.kind_task_available_list])
        # 基于action计算权重值并选取对应工序类型
        pri_rj_arr = action * pri_gap_arr + (1 - action) * pri_urg_arr  # 基于连续动作的综合次序值
        max_index = np.argmax(pri_rj_arr)
        rj = self.kind_task_available_list[max_index]
        return rj

    def machine_assignment(self, rj_selected):
        """机器分配规则"""
        machine_selectable_list = list(set(self.machine_idle_list)&set(self.kind_task_dict[rj_selected].machine_tuple))
        fluid_machine_selectable_list = list(set(self.machine_idle_list)&set(self.kind_task_dict[rj_selected].fluid_machine_list))
        if len(fluid_machine_selectable_list) == 0:
            m = min(machine_selectable_list, key=lambda x: self.time_mrj_dict[x][rj_selected])  # 选择加工时间最小的机器
        else:
            m = max(fluid_machine_selectable_list, key=lambda x: self.machine_dict[x].gap_rj_dict[rj_selected])  # 选择gap值最大的机器
        return m

    def compute_reward(self, weight_vector=None, completion=None, tardiness=None):
        """
        回报函数的选择:weight_vector(completion_weight, tardiness_weight)
        """
        function_selected = 4
        if completion is not None and tardiness is not None:
            return (self.completion_time_last - self.completion_time)/completion*weight_vector[0] + \
                   (self.delay_time_sum_last - self.delay_time_sum)/tardiness*weight_vector[1]
        elif completion is None and tardiness is None:
            if function_selected == 1:
                return self.completion_time_last - self.completion_time
            elif function_selected == 2:
                return self.delay_time_sum_last - self.delay_time_sum
            elif function_selected == 3:
                if self.delay_time_sum < self.delay_time_sum_last:
                    return 1
                elif self.delay_time_sum == self.delay_time_sum_last:
                    return 0
                else:
                    return -1
            elif function_selected == 4:
                if self.completion_time < self.completion_time_last:
                    return 1
                elif self.completion_time == self.completion_time_last:
                    return 0
                else:
                    return -1
            elif function_selected == 5:
                if self.completion_time <= self.completion_time_last * 1.0:
                    return 1
                elif self.completion_time == self.completion_time_last:
                    return 0
                else:
                    return -1
            elif weight_vector[1] == 1:
                return self.delay_time_sum_last - self.delay_time_sum
            elif weight_vector[0] == 1:
                return self.completion_time_last - self.completion_time
            else:
                raise MyError("未定义该回报函数")
        else:
            raise MyError("未定义该回报函数")

    """空闲机器列表"""
    @property
    def machine_idle_list(self):
        return [m for m in self.machine_tuple if self.machine_dict[m].state == 0]
    """可选加工工序类型列表"""
    @property
    def kind_task_available_list(self):
        return [(r, j) for (r, j) in self.kind_task_tuple if len(self.kind_task_dict[(r, j)].job_now_list) > 0 and
                set(self.kind_task_dict[(r, j)].machine_tuple) & set(self.machine_idle_list)]
    """流体解中可选工序集合"""
    @property
    def fluid_kind_task_available_list(self):
        return [(r, j) for (r, j) in self.kind_task_tuple if len(self.kind_task_dict[(r, j)].job_now_list) > 0 and
                set(self.kind_task_dict[(r, j)].fluid_machine_list) & set(self.machine_idle_list)]
    """机器的平均完工时间"""
    @property
    def ct_m_ave(self):
        return sum(machine_object.time_end for m, machine_object in self.machine_dict.items()) / self.machine_count
    """gap_ave"""
    @property
    def gap_ave_value(self):
        return sum(kind_task_object.gap for kind_task, kind_task_object in self.kind_task_dict.items())/len(self.kind_task_tuple)


# 测试环境
if __name__ == '__main__':
    DDT = 1.0
    M = 15
    S = 1
    file_name = 'DDT1.0_M15_R10'
    path = '/data/MPPPO'
    time_start = time.time()
    env_object = MO_FJSSP_Environment(use_instance=False, path=path, file_name=file_name)  # 定义环境对象
    state = env_object.reset()  # 初始化状态
    replay_list = []
    # 随机选择动作测试环境
    while not env_object.done:
        # action = (random.choice([0, 1, 2, 3, 4, 5]), random.choice([0, 1, 2, 3, 4]))
        action = 0.5
        # action = 1
        next_state, reward, done = env_object.step(action, [1, 0])
        replay_list.append([state, action, next_state, reward, done])
        state = next_state
        # print(env_object.machine_idle_list)
    print("累计回报:", env_object.reward_sum)
    print("总步数", env_object.step_count)
    # print("订单到达时间", [order_object.time_arrive for s, order_object in env_object.order_dict.items()])
    # print("订单交期时间", [order_object.time_delivery for s, order_object in env_object.order_dict.items()])
    print("机器完工时间", [machine_object.time_end for m, machine_object in env_object.machine_dict.items()])
    print("最大完工时间", env_object.completion_time)
    print("总延期时间：", env_object.delay_time_sum)
    # print("单周期耗时：", time.time() - time_start)

    # 画甘特图
    figure_object = FigGan(env_object)
    # figure_object.figure()



