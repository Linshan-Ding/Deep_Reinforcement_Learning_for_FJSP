import sys
from random import randint, uniform
import numpy as np, copy, csv
# 英文显示问题
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')  # 设置英文字体
# 中文显示问题
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="SimHei.ttf", size=12)  # 指定中文字体和字号
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
# 报错问题
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Instance():
    """
    问题特点
    文件命名方式 DDT0.5_M10_S1
    """
    def __init__(self, DDT, M, S):
        # 问题特点
        self.DDT = DDT  # 交期紧急度
        self.machine_count = M  # 机器数
        self.machine_tuple = tuple(m for m in range(self.machine_count))  # 机器元组
        self.order_count = S  # 新到达的订单总数
        self.order_tuple = tuple(s for s in range(self.order_count))  # 订单元组
        self.kind_count = randint(5, 15)  # 工件类型数
        self.kind_tuple = tuple(r for r in range(self.kind_count))  # 工件类型元组
        self.file_name = 'DDT' + str(DDT) + '_M' + str(M) + '_S' + str(S)  # 算例文件夹名
        # 工序元组索引，可选机器元组索引，在各机器上的加工时间，订单中各类型工件数量， 订单到达时间， 订单交期时间
        self.task_r_dict, self.machine_rj_dict, self.kind_task_m_dict, self.time_rjm_dict, self.count_sr_dict, \
        self.time_arrive_s_dict, self.time_delivery_s_dict, self.kind_task_tuple, self.time_mrj_dict = self.process_information()

    @property
    def J_r(self):
        """每种工件类型的工序数"""
        return randint(5, 10)
    @property
    def t_rjm(self):
        """工序在可选机器上的加工时间"""
        return randint(1, 20)
    @property
    def N_sr(self):
        """每个订单中包含的每种工件类型的工件数量"""
        return randint(5, 10)
    @property
    def t_si(self):
        """订单到达时间间隔"""
        return uniform(100, 200)

    def process_information(self):
        """生成实例文件数据，并存入csv文件"""
        # 机器、工件信息
        task_r_dict = {r: tuple(j for j in range(self.J_r)) for r in self.kind_tuple}  # [r]对应工序元组
        kind_task_tuple = tuple((r, j) for r in self.kind_tuple for j in task_r_dict[r])  # 工序类型元组
        machine_rj_dict = {r: {j: tuple(np.random.choice(self.machine_tuple, randint(1, self.machine_count), replace=False))
                               for j in task_r_dict[r]} for r in self.kind_tuple}  # [r][j]可选机器元组
        time_rjm_dict = {r: {j: {m: self.t_rjm for m in machine_rj_dict[r][j]}
                             for j in task_r_dict[r]} for r in self.kind_tuple}  # [r][j][m]加工时间
        kind_task_m_dict = {m: tuple((r, j) for r in self.kind_tuple for j in task_r_dict[r]
                                     if m in machine_rj_dict[r][j]) for m in self.machine_tuple}
        time_mrj_dict = {m: {rj: time_rjm_dict[rj[0]][rj[1]][m] for rj in kind_task_m_dict[m]} for m in self.machine_tuple}
        # 各工序加工时间均值
        time_rj_dict = {r: {j: sum([time_rjm_dict[r][j][m] for m in machine_rj_dict[r][j]]) / len(machine_rj_dict[r][j])
                            for j in task_r_dict[r]} for r in self.kind_tuple}
        # 订单信息
        count_sr_dict = {s: tuple(self.N_sr for r in range(len(task_r_dict))) for s in self.order_tuple}  # [s][r]工件类型的数量
        time_gap_s_dict = {s: sum([time_rj_dict[r][j] * count_sr_dict[s][r] for r in self.kind_tuple for j in task_r_dict[r]])
                              * self.DDT/(self.machine_count*2) for s in self.order_tuple}  # 各订单交期-到达时间差值
        time_interval_list = [self.t_si for s in range(self.order_count - 1)]  # 各订单的间隔时间
        time_interval_list.insert(0, 0)
        time_arrive_s_dict = {s: int(sum(time_interval_list[:s + 1])) for s in self.order_tuple}  # 各订单的到达时间
        time_delivery_list = [time_arrive_s_dict[s] + time_gap_s_dict[s] for s in self.order_tuple]
        time_delivery_list.sort()
        time_delivery_s_dict = {s: int(time_delivery_list[s]) for s in self.order_tuple}  # 各订单的交期时间

        return task_r_dict, machine_rj_dict, kind_task_m_dict, time_rjm_dict, count_sr_dict, time_arrive_s_dict, \
               time_delivery_s_dict, kind_task_tuple, time_mrj_dict

    def write_file(self):
        """写入csv文件"""
        file_path = 'DA3C'
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

# 生成算例
if __name__ == '__main__':
    # 程序需要确认是否覆盖原文件
    print("程序需要确认继续，请输入 y 继续(覆盖原生成文件，可导致数据丢失)或者 n 取消：")
    user_input = input()
    if user_input.lower() == 'y':
        print("继续执行程序...")
        # 初始化算例集参数
        DDT_list = [0.5, 1.0, 1.5]
        machine_count_list = [10, 15, 20]
        order_count_list = [1, 3, 5]
        file_name_list = []
        # 生成各实例文件
        for DDT in DDT_list:
            for M in machine_count_list:
                for S in order_count_list:
                    instance_object = Instance(DDT, M, S)
                    instance_object.write_file()
                    file_name_list.append(instance_object.file_name)
        print(file_name_list)
    else:
        print("取消执行程序。")
        sys.exit(0)

