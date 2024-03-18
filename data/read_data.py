# -*- coding: utf-8 -*-
import csv, os, re

class Data():
    def __init__(self, path, file_name):
        self.file_name = file_name  # 算例文件名
        self.csv_name_list = ['based_data.csv', 'process_data.csv', 'order_data.csv']
        self.path = path  # 路径
        # 基础数据信息 工件类型数， 机器数， 订单数
        self.kind_count, self.machine_count, self.order_count = self.read(self.csv_name_list[0])
        self.machine_tuple = tuple(m for m in range(self.machine_count))  # 机器元组
        self.order_tuple = tuple(s for s in range(self.order_count))  # 订单元组
        self.kind_tuple = tuple(r for r in range(self.kind_count))  # 工件类型元组
        # 订单信息 订单中各类型工件数量， 订单到达时间， 订单交期时间
        self.count_sr_dict, self.time_arrive_s_dict, self.time_delivery_s_dict = self.read(self.csv_name_list[2])
        # 加工信息 工序元组索引，可选机器元组索引，在各机器上的加工时间
        self.task_r_dict, self.machine_rj_dict, self.time_rjm_dict = self.read(self.csv_name_list[1])

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
            return kind_count, machine_count, order_count
        elif csv_name == 'process_data.csv':
            task_r_dict = {kind: [] for kind in self.kind_tuple}
            machine_rj_dict = {kind: {} for kind in self.kind_tuple}
            time_rjm_dict = {kind: {} for kind in self.kind_tuple}
            for row in rows[1:]:
                kind = self.str_int(row[0])
                task = self.str_int(row[1])
                machine_str = self.str_int_tuple(row[2])
                time_str = self.str_int_tuple(row[3])
                task_r_dict[kind].append(task)
                machine_rj_dict[kind][task] = machine_str
                time_rjm_dict[kind][task] = time_str
            # 预处理数据
            for key, value in task_r_dict.items():
                task_r_dict[key] = tuple(value)
            time_rjm_dict = {kind: {task: {machine: time for machine, time in zip(machine_rj_dict[kind][task], time_rjm_dict[kind][task])}
                                    for task in task_r_dict[kind]} for kind in self.kind_tuple}

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
    file_name = 'DDT0.5_M10_S5'
    path = '../data/generated'
    data = Data(path, file_name)
    print(data)






