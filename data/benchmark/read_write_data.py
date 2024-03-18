"""
读取标准算例文件并生成算法可读取文件夹
"""
import sys
from random import randint, uniform
import numpy as np, copy, csv
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class DataRead():
    """文件读取类"""
    def __init__(self, path, file_name):
        self.path = path   # 文件存储路径
        self.file_name = file_name  # 文件名
        # 问题特点
        self.DDT = 1.0  # 交期紧急度
        self.machine_count = None  # 机器数
        self.machine_tuple = None  # 机器元组
        self.order_count = 1  # 新到达的订单总数
        self.order_tuple = tuple(s for s in range(self.order_count))  # 订单元组
        self.kind_count = None  # 工件类型数
        self.kind_tuple = None  # 工件类型元组
        # 工序元组索引，可选机器元组索引，在各机器上的加工时间，订单中各类型工件数量， 订单到达时间， 订单交期时间
        self.task_r_dict, self.machine_rj_dict, self.kind_task_m_dict, self.time_rjm_dict, self.count_sr_dict, \
        self.time_arrive_s_dict, self.time_delivery_s_dict, self.kind_task_tuple, self.time_mrj_dict = self.read_data()

    def read_data(self):
        file = open(self.path + '/{}.fjs'.format(self.file_name))
        data = file.readlines()  # 按行读取数据（每一行存储为一个字符串）
        file.close()
        data_list = [[] for line in data]
        for i in range(len(data)):
            line_list = list(data[i])
            number_char = ''
            for number in line_list:
                if number not in ['\t', '', ' ', '\n']:  # 数字间的字符
                    number_char += number  # 数字连接
                    continue
                else:
                    if number_char != '':
                        number_char = float(number_char)
                        if number_char % 1 == 0:  # 若是整数则转换为整数类型
                            number_char = int(number_char)
                        else:
                            print('存在非整数')
                        data_list[i].append(number_char)
                    number_char = ''
        # print('初步处理后的数据:\n', data_list)
        # 清除空列表[]
        for i in range(len(data_list)):
            if len(data_list[i]) == 0:
                data_list.remove([])
        machine_count = data_list[0][1]  # 机器数量
        self.machine_count = machine_count
        self.machine_tuple = tuple(m for m in range(self.machine_count))
        kind_count = data_list[0][0]  # 工件种类数量
        self.kind_count = kind_count
        self.kind_tuple = tuple(r for r in range(self.kind_count))
        task_kind = []  # 每种工件的工序数
        for job in data_list[1:]:
            task_kind.append(int(job[0]))
        machine_list = range(machine_count)  # 机器列表
        number_classes = sum(task_kind)  # 类总数
        classes_list = range(number_classes)  # 流体类列表
        # 各类可选加工机器列表machine_class[0]=[0, 2, 3]类0可在机器0，2，3上加工
        machine_class = [[] for i in range(number_classes)]
        # 各类在各机器上的加工时间列表和machine_class对应
        time_machine_class = [[] for i in range(number_classes)]
        # 每个类的可选加工机器和对应加工时间列表
        class_machine_time = []  # 初始化
        for job in data_list[1:]:
            task_number = job[0]
            job = job[1:]
            i = 0
            k = 0
            while k < task_number:
                machine_set_number = job[i]
                class_machine_time.append(job[i + 1:i + 1 + machine_set_number * 2])
                i = i + 1 + machine_set_number * 2
                k += 1
        # print("每个类的可选加工机器+时间", class_machine_time)
        # print("类数量", len(class_machine_time))
        # 生成各类可选加工机器和对应加工时间列表
        for k in classes_list:
            for i in range(len(class_machine_time[k])):
                if i % 2 == 0:
                    machine_class[k].append(class_machine_time[k][i] - 1)
                else:
                    time_machine_class[k].append(class_machine_time[k][i])
        # 各机器可选择的加工类和该类在机器上的初始切换时间
        class_machine = [[] for i in range(machine_count)]
        for m in machine_list:
            for k in classes_list:
                if m in machine_class[k]:
                    class_machine[m].append(k)
        # 由工序到类的索引
        classes_kind = [[i for i in range((sum(task_kind[0:j]) - task_kind[j - 1]), sum(task_kind[0:j]))] for j in range(1, kind_count + 1)]
        # 由类到工序的索引
        kind_task_class = [[i, j] for k in classes_list for i in range(kind_count) for j in range(task_kind[i]) if classes_kind[i][j] == k]
        kind_task_class_dict = {k: tuple(kind_task_class[k]) for k in classes_list}  # 由类号-(r, j)的索引
        class_kind_task_dict = {value: key for key, value in kind_task_class_dict.items()}

        # 生成写入数据
        # 机器、工件信息
        task_r_dict = {r: tuple(j for j in range(task_kind[r])) for r in self.kind_tuple}  # [r]对应工序元组
        kind_task_tuple = tuple((r, j) for r in self.kind_tuple for j in task_r_dict[r])  # 工序类型元组
        machine_rj_dict = {r: {j: tuple(machine_class[class_kind_task_dict[(r, j)]]) for j in task_r_dict[r]} for r in self.kind_tuple}  # [r][j]可选机器元组
        # time_rjm_dict = {}
        # for r in self.kind_tuple:
        #     time_rjm_dict[r] = {}
        #     for j in task_r_dict[r]:
        #         time_rjm_dict[r][j] = {}
        #         k = class_kind_task_dict[(r, j)]  # 类号
        #         for m in machine_rj_dict[r][j]:
        #             index = machine_class[k].index(m)
        #             time_rjm_dict[r][j][m] = time_machine_class[k][index]
        time_rjm_dict = {r: {j: {m: time_machine_class[class_kind_task_dict[(r, j)]][machine_class[class_kind_task_dict[(r, j)]].index(m)] for m in machine_rj_dict[r][j]} for j in task_r_dict[r]} for r in self.kind_tuple}  # [r][j][m]加工时间
        kind_task_m_dict = {m: tuple((r, j) for r in self.kind_tuple for j in task_r_dict[r] if m in machine_rj_dict[r][j]) for m in self.machine_tuple}
        time_mrj_dict = {m: {rj: time_rjm_dict[rj[0]][rj[1]][m] for rj in kind_task_m_dict[m]} for m in self.machine_tuple}
        # 各工序加工时间均值
        time_rj_dict = {r: {j: sum([time_rjm_dict[r][j][m] for m in machine_rj_dict[r][j]]) / len(machine_rj_dict[r][j]) for j in task_r_dict[r]} for r in self.kind_tuple}
        # 订单信息
        count_sr_dict = {s: tuple(1 for r in range(len(task_r_dict))) for s in self.order_tuple}  # [s][r]工件类型的数量
        time_gap_s_dict = {s: sum([time_rj_dict[r][j] * count_sr_dict[s][r] for r in self.kind_tuple for j in task_r_dict[r]]) * self.DDT / (self.machine_count * 2) for s in self.order_tuple}  # 各订单交期-到达时间差值
        time_interval_list = [0 for s in range(self.order_count - 1)]  # 各订单的间隔时间
        time_interval_list.insert(0, 0)
        time_arrive_s_dict = {s: int(sum(time_interval_list[:s + 1])) for s in self.order_tuple}  # 各订单的到达时间
        time_delivery_list = [time_arrive_s_dict[s] + time_gap_s_dict[s] for s in self.order_tuple]
        time_delivery_list.sort()
        time_delivery_s_dict = {s: int(time_delivery_list[s]) for s in self.order_tuple}  # 各订单的交期时间

        return task_r_dict, machine_rj_dict, kind_task_m_dict, time_rjm_dict, count_sr_dict, time_arrive_s_dict, \
               time_delivery_s_dict, kind_task_tuple, time_mrj_dict

    def write_file(self):
        """写入csv文件"""
        file_path = self.path
        os.makedirs(os.path.join(file_path, self.file_name), exist_ok=True)  # 新建实例文件夹
        file_csv = {'based_data.csv': ['kind_count', 'machine_count', 'order_count', 'DDT'],
                    'process_data.csv': ['kind', 'task', 'machine_selectable', 'process_time'],
                    'order_data.csv': ['order', 'time_arrive', 'time_delivery', 'kind_number']}

        for csv_name, header in file_csv.items():
            data_file = os.path.join(file_path, self.file_name, csv_name)
            with open(data_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                rows = []  # 初始化写入数据
                if csv_name == 'based_data.csv':
                    rows.append([self.kind_count, self.machine_count, self.order_count, self.DDT])
                elif csv_name == 'process_data.csv':
                    for r in self.kind_tuple:
                        for j in self.task_r_dict[r]:
                            time_machine_tuple = tuple(self.time_rjm_dict[r][j][m] for m in self.machine_rj_dict[r][j])
                            rows.append([r, j, self.machine_rj_dict[r][j], time_machine_tuple])
                else:
                    for s in self.order_tuple:
                        rows.append([s, self.time_arrive_s_dict[s], self.time_delivery_s_dict[s], self.count_sr_dict[s]])
                writer.writerows(rows)
        print("写入完成")

# 测试
if __name__ == '__main__':
    path = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/data/benchmark/Brandimarte_Data'
    file_name_list = ['Mk01', 'Mk02', 'Mk03', 'Mk04', 'Mk05', 'Mk06', 'Mk07', 'Mk08', 'Mk09', 'Mk10']
    # file_name_list = ['la01', 'la40']
    for file_name in file_name_list:
        read_object = DataRead(path=path, file_name=file_name)
        read_object.write_file()
        print('写入文件：', file_name)
