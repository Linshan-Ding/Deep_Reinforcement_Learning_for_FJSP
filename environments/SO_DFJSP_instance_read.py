# -*- coding: utf-8 -*-
import copy, ast
import csv, os, re


class Data():
    def __init__(self, path=None, file_name=None):
        self.file_name = file_name  # 算例文件名
        self.csv_name_list = ['based_data.csv', 'process_data.csv', 'order_data.csv']
        self.path = path  # 路径
        # 基础数据信息 工件类型数， 机器数， 订单数, 延期度
        self.kind_count, self.machine_count, self.order_count, self.DDT = self.read(self.csv_name_list[0])
        self.machine_tuple = tuple(m for m in range(self.machine_count))  # 机器元组
        self.order_tuple = tuple(s for s in range(self.order_count))  # 订单元组
        self.kind_tuple = tuple(r for r in range(self.kind_count))  # 工件类型元组
        # 订单信息 订单中各类型工件数量， 订单到达时间， 订单交期时间
        self.count_sr_dict, self.time_arrive_s_dict, self.time_delivery_s_dict = self.read(self.csv_name_list[2])
        # 各机器的待机功率
        # 加工信息 工序元组索引，可选机器元组索引，在各机器上的加工时间
        self.task_r_dict, self.machine_rj_dict, self.time_rjm_dict = self.read(self.csv_name_list[1])
        self.kind_task_m_dict, self.kind_task_tuple, self.time_mrj_dict, self.time_rj_dict = self.process()

    def process(self):
        """生成额外索引"""
        kind_task_tuple = tuple((r, j) for r in self.kind_tuple for j in self.task_r_dict[r])  # 工序类型元组
        kind_task_m_dict = {m: tuple((r, j) for (r, j) in kind_task_tuple if m in self.machine_rj_dict[(r, j)]) for m in self.machine_tuple}
        time_mrj_dict = {m: {(r, j): self.time_rjm_dict[(r, j)][m] for (r, j) in kind_task_m_dict[m]} for m in self.machine_tuple}
        time_rj_dict = {(r, j): sum([time_mrj_dict[m][(r, j)] for m in self.machine_rj_dict[(r, j)]]) / len(self.machine_rj_dict[(r, j)]) for (r, j) in kind_task_tuple}
        return kind_task_m_dict, kind_task_tuple, time_mrj_dict, time_rj_dict

    def str_int_tuple(self, s):
        """字符串中提取数字并转为int类型元组"""
        nums = [int(i) for i in re.findall(r'\d+', s)]
        return tuple(nums)

    def str_int(self, s):
        """字符串中提取数字并转为int类型"""
        nums = [int(i) for i in re.findall(r'\d+', s)]
        return nums[0]

    def read(self, csv_name):
        """读取文件数据"""
        data_file = os.path.join(self.path, self.file_name, csv_name)
        with open(data_file, 'r') as f:
            reader = csv.reader(f)
            rows = []
            for row in reader:
                rows.append(row)
        if csv_name == 'based_data.csv':
            kind_count = self.str_int(rows[1][0])
            machine_count = self.str_int(rows[1][1])
            order_count = self.str_int(rows[1][2])
            DDT = self.str_int(rows[1][3])
            return kind_count, machine_count, order_count, DDT
        elif csv_name == 'process_data.csv':
            task_r_dict = {kind: [] for kind in self.kind_tuple}
            machine_rj_dict = {}
            time_rjm_dict = {}
            for row in rows[1:]:
                kind = self.str_int(row[0])
                task = self.str_int(row[1])
                machine_str = self.str_int_tuple(row[2])
                time_str = self.str_int_tuple(row[3])
                task_r_dict[kind].append(task)
                machine_rj_dict[(kind, task)] = machine_str
                time_rjm_dict[(kind, task)] = time_str
            # 预处理数据
            for key, value in task_r_dict.items():
                task_r_dict[key] = tuple(value)
            for kind in self.kind_tuple:
                for task in task_r_dict[kind]:
                    time_machine = {}
                    for machine, time in zip(machine_rj_dict[(kind, task)], time_rjm_dict[(kind, task)]):
                        time_machine[machine] = time
                    time_rjm_dict[(kind, task)] = copy.deepcopy(time_machine)
            return task_r_dict, machine_rj_dict, time_rjm_dict
        else:
            count_sr_dict = {}
            time_arrive_s_dict = {}
            time_delivery_s_dict = {}
            for row in rows[1:]:
                order = self.str_int(row[0])
                time_arrive = self.str_int(row[1])
                time_delivery = self.str_int(row[2])
                kind_count = self.str_int_tuple(row[3])
                count_sr_dict[order] = kind_count
                time_arrive_s_dict[order] = time_arrive
                time_delivery_s_dict[order] = time_delivery
            return count_sr_dict, time_arrive_s_dict, time_delivery_s_dict


# 测试
if __name__ == '__main__':
    file_name = 'DDT0.5_M10_S1'

    # path = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/data/DA3C'
    # path_write = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/results/DA3C/task_number.csv'
    # file_name_list \
    #     = ['DDT0.5_M10_S1', 'DDT0.5_M10_S3', 'DDT0.5_M10_S5', 'DDT0.5_M15_S1', 'DDT0.5_M15_S3', 'DDT0.5_M15_S5',
    #        'DDT0.5_M20_S1', 'DDT0.5_M20_S3', 'DDT0.5_M20_S5', 'DDT1.0_M10_S1', 'DDT1.0_M10_S3', 'DDT1.0_M10_S5',
    #        'DDT1.0_M15_S1', 'DDT1.0_M15_S3', 'DDT1.0_M15_S5', 'DDT1.0_M20_S1', 'DDT1.0_M20_S3', 'DDT1.0_M20_S5',
    #        'DDT1.5_M10_S1', 'DDT1.5_M10_S3', 'DDT1.5_M10_S5', 'DDT1.5_M15_S1', 'DDT1.5_M15_S3', 'DDT1.5_M15_S5',
    #        'DDT1.5_M20_S1', 'DDT1.5_M20_S3', 'DDT1.5_M20_S5']

    # 多目标静态工序数读取
    path = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/data/MPPPO'
    path_write = '/results/MPPPO/common_rules/task_number.csv'
    file_name_list \
        = ['DDT0.5_M10_R5', 'DDT0.5_M10_R10', 'DDT0.5_M10_R15', 'DDT0.5_M15_R5', 'DDT0.5_M15_R10', 'DDT0.5_M15_R15', 'DDT0.5_M20_R5', 'DDT0.5_M20_R10',
            'DDT0.5_M20_R15', 'DDT1.0_M10_R5', 'DDT1.0_M10_R10', 'DDT1.0_M10_R15', 'DDT1.0_M15_R5', 'DDT1.0_M15_R10', 'DDT1.0_M15_R15', 'DDT1.0_M20_R5',
            'DDT1.0_M20_R10', 'DDT1.0_M20_R15', 'DDT1.5_M10_R5', 'DDT1.5_M10_R10', 'DDT1.5_M10_R15', 'DDT1.5_M15_R5', 'DDT1.5_M15_R10', 'DDT1.5_M15_R15',
            'DDT1.5_M20_R5', 'DDT1.5_M20_R10', 'DDT1.5_M20_R15']

    for file_name in file_name_list:
        data = Data(path, file_name)
        # 计算工序总数并写入文件
        task_total_number = 0
        for s in data.order_tuple:
            for r in data.kind_tuple:
                task_total_number += data.count_sr_dict[s][r] * len(data.task_r_dict[r])
        print(file_name + ':', task_total_number)
        # 写入文件
        with open(path_write, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [file_name, task_total_number]
            writer.writerow(row)
