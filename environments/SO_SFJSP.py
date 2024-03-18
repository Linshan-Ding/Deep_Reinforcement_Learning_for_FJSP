"""
单目标柔性作业车间调度环境+离散动作
目标：最大完工时间
"""
import random, math, time
import numpy as np
from environments.SO_FJSSP import SO_FJSSP_Environment
from utilities.Utility_Class import FigGan, MyError


class SO_SFJSP_Environment(SO_FJSSP_Environment):
    """多目标柔性作业车间静态调度环境"""
    def __init__(self, use_instance=True, **kwargs):
        super().__init__(use_instance=use_instance, **kwargs)
        self.state_size = 18  # 状态空间
        self.action_types = "DISCRETE"  # 动作类型
        self.observation_space = 9  # 观察的动态状态向量空间
        self.static_state_space = 0  # 静态状态向量空间
        self.action_space = 20
        self.order_object = self.order_dict[0]  # 实例订单对象
        self.static_state = self.static_state_extract()  # 静态特征
        self.reward_sum = 0  # 累计回报
        self.completion_time = 0  # 当前时间点完工时间
        self.completion_time_last = 0  # 上一时间步完工时间
        self.actions = tuple([(task_rule, machine_rule) for task_rule in range(4) for machine_rule in range(5)])

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
        self.state = np.concatenate((self.observation_state, self.state_gap), axis=0)  # 状态向量 [v(t), v(t) - v(t-1)]
        self.next_state = None  # 下一步状态
        self.reward = None  # 即时奖励
        self.done = False  # 是否为终止状态
        return self.state

    def static_state_extract(self):
        """提取任务相关静态状态"""
        M = self.machine_count  # 机器数
        R = self.kind_count  # 工件类型数
        N_ave = sum(self.order_object.count_kind[r] for r in self.kind_tuple)/self.kind_count  # 各工件类型的工件数均值
        N_std = math.sqrt(sum(math.pow(self.order_object.count_kind[r] - N_ave, 2) for r in self.kind_tuple)/self.kind_count)  # 各工件类型的工件数标准差
        J_ave = sum(len(self.task_r_dict[r]) for r in self.kind_tuple)/self.kind_count  # 工件类型的工序数均值
        J_std = math.sqrt(sum(math.pow(len(self.task_r_dict[r]) - J_ave, 2) for r in self.kind_tuple)/self.kind_count)
        return np.array([M, R, N_ave, N_std, J_ave, J_std])

    def state_extract(self):
        """提取状态向量更新相关参数"""
        M_idle_ratio = len(self.machine_idle_list) / self.machine_count  # 空闲机器数占总机器数的比例
        ct_m_ave = self.ct_m_ave  # 机器的平均完工时间
        ct_m_std = math.sqrt(sum(math.pow(machine_object.time_end - ct_m_ave, 2) for m, machine_object
                                 in self.machine_dict.items()) / self.machine_count)  # 机器完工时间标准差
        ratio_idle = len(self.fluid_kind_task_available_list) / (len(self.kind_task_available_list) + 1e-08)  # 可选工序类型数比值
        cro_ave = sum(kind_task_object.finish_rate for (r, j), kind_task_object in
                      self.kind_task_dict.items()) / len(self.kind_task_tuple)  # 工序类型完工率均值
        cro_std = math.sqrt(sum(math.pow(kind_task_object.finish_rate - cro_ave, 2) for (r, j), kind_task_object
                                in self.kind_task_dict.items()) / len(self.kind_task_tuple))  # 工序类型完工率标准差
        gap_ave = sum(kind_task_object.gap_rate for (r, j), kind_task_object
                      in self.kind_task_dict.items()) / len(self.kind_task_tuple)  # 工序类型gap_rj均值
        gap_std = math.sqrt(sum(math.pow(kind_task_object.gap_rate - gap_ave, 2) for (r, j), kind_task_object
                                in self.kind_task_dict.items()) / len(self.kind_task_tuple))  # 工序类型gap_rj标准差
        gap_m_ave = sum(machine_object.gap_ave for m, machine_object in self.machine_dict.items()) / self.machine_count  # 机器gap_m均值
        gap_m_std = math.sqrt(sum(math.pow(machine_object.gap_ave - gap_m_ave, 2) for m, machine_object
                                  in self.machine_dict.items()) / self.machine_count)  # 机器gap_m标准差

        return np.array([M_idle_ratio, ct_m_std, cro_ave, cro_std, ratio_idle, gap_ave, gap_std, gap_m_ave, gap_m_std])

    def step(self, action):
        """根据动作选择工序选择规则+机器分配规则"""
        task_rule = self.actions[action][0] + 1  # 工序类型选择规则
        machine_rule = self.actions[action][1] + 1  # 机器选择规则
        rj_selected = self.task_select(task_rule)  # 选择的工序类型
        m_selected = self.machine_select(machine_rule, rj_selected)  # 选择的机器
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
        self.next_state = np.concatenate((self.observation_state, self.state_gap), axis=0)
        self.delay_time_sum = self.delay_time_sum_processed + self.delay_time_sum_unprocessed  # 实际总延期时间
        self.reward = self.compute_reward()  # 即时奖励
        # print("即时回报值：", self.reward)
        self.reward_sum += self.reward  # 更新累计回报
        self.delay_time_sum_last = self.delay_time_sum  # 更新上一时间步实际总的延期时间
        self.completion_time_last = self.completion_time  # 更新上一步完工时间
        self.state = self.next_state
        return self.state, self.reward, self.done

    def task_select(self, task_rule):
        """6个工序选择规则"""
        if task_rule == 1:  # 工序选择规则1
            if len(self.fluid_kind_task_available_list) == 0:
                rj = max(self.kind_task_available_list, key=lambda x: self.kind_task_dict[x].gap)
            else:
                rj = max(self.fluid_kind_task_available_list, key=lambda x: self.kind_task_dict[x].gap)
        elif task_rule == 2:  # 工序选择规则4
            if len(self.fluid_kind_task_available_list) == 0:
                rj = min(self.kind_task_available_list, key=lambda x: self.time_min_rj(x))
            else:
                rj = min(self.fluid_kind_task_available_list, key=lambda x: self.time_min_fluid_rj(x))
        elif task_rule == 3:   # 工序选择规则5
            rj = min(self.kind_task_available_list, key=lambda x: self.time_min_rj(x))
        elif task_rule == 4:  # 工序选择规则6
            rj = random.choice(self.kind_task_available_list)
        else:
            raise MyError("报错：未定义该工序动作规则")
        return rj

    def machine_select(self, machine_rule, rj_selected):
        """4个机器分配规则"""
        machine_selectable_list = list(set(self.machine_idle_list)&set(self.kind_task_dict[rj_selected].machine_tuple))
        fluid_machine_selectable_list = list(set(self.machine_idle_list)&set(self.kind_task_dict[rj_selected].fluid_machine_list))
        if machine_rule == 1:  # 机器分配规则1
            if len(fluid_machine_selectable_list) == 0:
                m = max(machine_selectable_list, key=lambda x: self.machine_dict[x].gap_rj_dict[rj_selected])
            else:
                m = max(fluid_machine_selectable_list, key=lambda x: self.machine_dict[x].gap_rj_dict[rj_selected])
        elif machine_rule == 2:  # 机器分配规则2
            if len(fluid_machine_selectable_list) == 0:
                m = min(machine_selectable_list, key=lambda x: self.time_mrj_dict[x][rj_selected])
            else:
                m = min(fluid_machine_selectable_list, key=lambda x: self.time_mrj_dict[x][rj_selected])
        elif machine_rule == 3:  # 机器分配规则3
            m = min(machine_selectable_list, key=lambda x: self.time_mrj_dict[x][rj_selected])
        elif machine_rule == 4:  # 机器分配规则4
            if len(fluid_machine_selectable_list) == 0:
                m = max(machine_selectable_list, key=lambda x: self.machine_dict[x].gap_ave)
            else:
                m = max(fluid_machine_selectable_list, key=lambda x: self.machine_dict[x].gap_ave)
        elif machine_rule == 5:  # 机器分配规则5
            m = random.choice(machine_selectable_list)
        else:
            raise MyError("报错：未定义该机器分配规则。")
        return m

    def compute_reward(self):
        """回报函数的选择"""
        reward_function = 1
        if reward_function == 1:
            return -(self.completion_time - self.completion_time_last)/self.fluid_completed_time
        else:
            return -(self.completion_time - self.completion_time_last)

    def machine_available_rj_selectable(self, rj_selected):
        """返回选择的工序实际可选加工机器集合"""
        machine_selectable_list = list(set(self.machine_idle_list) & set(self.kind_task_dict[rj_selected].machine_tuple))
        return machine_selectable_list

    def machine_fluid_available_rj_selectable(self, rj_selected):
        """返回选择的工序流体可选加工机器集合"""
        fluid_machine_selectable_list = list(set(self.machine_idle_list) & set(self.kind_task_dict[rj_selected].fluid_machine_list))
        return fluid_machine_selectable_list

    def time_min_rj(self, rj):
        """工序类在制造系统中空闲机器上的最小加工时间"""
        machine_selectable_list = self.machine_available_rj_selectable(rj)
        machine_min_time = min(machine_selectable_list, key=lambda x: self.time_mrj_dict[x][rj])
        return self.time_mrj_dict[machine_min_time][rj]

    def time_min_fluid_rj(self, rj):
        """工序类在流体解中空闲机器上的最小加工时间"""
        machine_selectable_list = self.machine_fluid_available_rj_selectable(rj)
        machine_min_time = min(machine_selectable_list, key=lambda x: self.time_mrj_dict[x][rj])
        return self.time_mrj_dict[machine_min_time][rj]

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




