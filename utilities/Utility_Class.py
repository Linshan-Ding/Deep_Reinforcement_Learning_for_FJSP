"""
通用类
"""
import os
import matplotlib.pyplot as plt, sys, openpyxl, numpy as np, copy, csv, os, re, pickle, mpl_toolkits.mplot3d.axes3d as axes3d
from random import randint, uniform
from matplotlib.ticker import ScalarFormatter
from openpyxl.styles import numbers
from scipy.spatial.distance import cdist
from matplotlib.font_manager import FontProperties
import matplotlib as mpl, pandas as pd, matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties  # 中文显示问题
# 设置显示问题
font = FontProperties(fname="SimHei.ttf", size=12)  # 指定中文字体和字号
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
mpl.rcParams['font.family'] = 'Times New Roman'  # 设置全局字体为Times New Roman
plt.rc('font', family='Times New Roman')  # 设置英文字体# 英文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题


class PlotFigure():
    """画图类"""
    def __init__(self):
        self.figure_name = ['帕累托前沿图', '甘特图', '收敛曲线图']

    def Pareto_figure(self, solutions_dict, algorithms, instance, algorithm_names=None, local='upper left'):
        """画帕累托图"""
        color_symbols = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']
        marker_symbols = ['*', 'o', 's', '<', 'D', 'p', 'h', 'x', '+', 'v', '^', '>']
        fig, ax = plt.subplots()
        for algorithm in algorithms:
            indexed = algorithms.index(algorithm)
            algorithm_data = np.array(solutions_dict[(algorithm, instance)])
            if algorithm_names is None:
                ax.scatter(algorithm_data[:, 0], algorithm_data[:, 1], c=color_symbols[indexed],
                           marker=marker_symbols[indexed], label=algorithm)
            else:
                algorithm_index = algorithms.index(algorithm)
                ax.scatter(algorithm_data[:, 0], algorithm_data[:, 1], c=color_symbols[indexed],
                           marker=marker_symbols[indexed], label=algorithm_names[algorithm_index])

        # 添加标题、轴标题和图例
        ax.set_title('Pareto Front')
        ax.set_xlabel('Makespan')
        ax.set_ylabel('Total tardiness')
        ax.legend(prop=FontProperties(family='Times New Roman'), loc=local)
        plt.title(instance)
        plt.tight_layout()
        plt.savefig(instance + 'pareto_front.svg', format='svg')

    def Pareto_figure_2D(self, solutions_dict, algorithms, instance, objectives, algorithm_names=None, local='upper left'):
        """画帕累托图"""
        objectives_list = ['Makespan', 'Tardiness', 'Energy']
        color_symbols = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']
        marker_symbols = ['*', 'o', 's', '<', 'D', 'p', 'h', 'x', '+', 'v', '^', '>']
        fig, ax = plt.subplots()
        for algorithm in algorithms:
            indexed = algorithms.index(algorithm)
            algorithm_data = np.array(solutions_dict[(algorithm, instance)])
            if algorithm_names is None:
                ax.scatter(algorithm_data[:, objectives[0]], algorithm_data[:, objectives[1]], c=color_symbols[indexed],
                           marker=marker_symbols[indexed], label=algorithm)
            else:
                algorithm_index = algorithms.index(algorithm)
                ax.scatter(algorithm_data[:, objectives[0]], algorithm_data[:, objectives[1]], c=color_symbols[indexed],
                           marker=marker_symbols[indexed], label=algorithm_names[algorithm_index])

        # 添加标题、轴标题和图例
        ax.set_title('Pareto Front')
        ax.set_xlabel(objectives_list[objectives[0]])
        ax.set_ylabel(objectives_list[objectives[1]])
        ax.legend(prop=FontProperties(family='Times New Roman'), loc=local)
        plt.title(instance)
        plt.savefig(instance + '_front_{}{}.svg'.format(objectives[0], objectives[1]), format='svg')

    def Pareto_figure_3D(self, solutions_dict, algorithms, instance, algorithm_names=None, local='upper left'):
        """画帕累托图"""
        color_symbols = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray']
        marker_symbols = ['*', 'o', 's', '<', 'D', 'p', 'h', 'x', '+', 'v', '^', '>']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for algorithm in algorithms:
            indexed = algorithms.index(algorithm)
            algorithm_data = np.array(solutions_dict[(algorithm, instance)])
            if algorithm_names is None:
                ax.scatter(algorithm_data[:, 0], algorithm_data[:, 1], algorithm_data[:, 2], c=color_symbols[indexed],
                           marker=marker_symbols[indexed], label=algorithm)
            else:
                algorithm_index = algorithms.index(algorithm)
                ax.scatter(algorithm_data[:, 0], algorithm_data[:, 1], algorithm_data[:, 2], c=color_symbols[indexed],
                           marker=marker_symbols[indexed], label=algorithm_names[algorithm_index])

        # 添加标题、轴标题和图例
        ax.set_title('Pareto Front')
        ax.set_xlabel('Makespan')
        ax.set_ylabel('Tardiness')
        ax.set_zlabel('Energy')
        ax.legend(prop=FontProperties(family='Times New Roman'), loc=local)
        # 设置显示
        ax.xaxis._axinfo["color"] = (0.925, 0.125, 0.90, 0.25)
        ax.xaxis.pane.set_color("none")
        ax.yaxis.pane.set_color("none")
        ax.zaxis.pane.set_color("none")
        ax.xaxis._axinfo["grid"].update({"linewidth": .3, "color": "gray"})
        ax.yaxis._axinfo["grid"].update({"linewidth": .3, "color": "gray"})
        ax.zaxis._axinfo["grid"].update({"linewidth": .3, "color": "gray"})
        # 设置Z轴刻度位置
        ax.zaxis._axinfo['juggled'] = (1, 2, 0)
        # 使用 ScalarFormatter 来设置科学计数法
        formatter = ScalarFormatter(useMathText=True)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.zaxis.set_major_formatter(formatter)
        # 添加图标题
        plt.title(instance)
        plt.savefig(instance + 'pareto.svg', format='svg', bbox_inches='tight')


class DataProcess:
    """数据处理类"""
    def __init__(self):
        self.function = ['计算帕累托解', 'GD', 'IGD', 'Spread', '写入pickle', '读取pickle']

    def normalized_data(self, data, max_values):
        """根据各目标最大值归一化数据"""
        max_values = self.to_numpy_array(max_values)
        data = self.to_numpy_array(data)
        # 对每个元素进行归一化操作
        normalized_data = data / max_values
        return normalized_data

    def filter_pareto_front(self, population):
        """返回解集中的帕累托前沿解"""
        population = self.to_numpy_array(population)
        pareto_front = []
        n = len(population)
        for i in range(n):
            dominated = False
            for j in range(n):
                if all(population[j] <= population[i]) and any(population[j] < population[i]):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(population[i])
        pareto_front = np.unique(pareto_front, axis=0)
        return pareto_front

    def pareto_count_delete_one(self, solution, solutions_all):
        """
        :param solution: 删除的解
        :param solutions_all: 解集
        :return: 删除一个解后的非支配解数量
        """
        solutions_all.remove(solution)
        solutions_pareto = self.filter_pareto_front(solutions_all).tolist()
        return len(solutions_pareto)

    def to_numpy_array(self, lst):
        if isinstance(lst, np.ndarray):
            return lst
        else:
            return np.array(lst)

    def distance_square(self, point1, point2):
        """计算两个点之间的欧氏距离"""
        return np.sum((point1 - point2) ** 2)

    def calculate_igd(self, true_front, population):
        """计算 IGD 值"""
        true_front = self.to_numpy_array(true_front)
        population = self.to_numpy_array(population)
        igd = 0.0
        for true_point in true_front:
            min_distance = np.inf
            for point in population:
                distance = self.distance_square(true_point, point)
                distance = np.sqrt(distance)
                if distance < min_distance:
                    min_distance = distance
            igd += min_distance
        igd /= len(true_front)
        return igd

    def calculate_gd(self, true_front, population):
        """计算 GD 值"""
        true_front = self.to_numpy_array(true_front)
        population = self.to_numpy_array(population)
        gd = 0.0
        for point in population:
            min_distance = np.inf
            for true_point in true_front:
                distance = self.distance_square(point, true_point)
                if distance < min_distance:
                    min_distance = distance
            gd += min_distance
        gd = np.sqrt(gd)
        gd /= len(population)
        return gd

    def calculate_spread(self, population, true_front):
        """计算 Spread 值"""
        true_front = self.to_numpy_array(true_front)
        population = self.to_numpy_array(population)
        diaa = []
        for point1 in population:
            min_distance = np.inf
            for point2 in population:
                distance = self.distance_square(point1, point2)
                distance = np.sqrt(distance)
                if distance < min_distance and distance != 0.0:
                    min_distance = distance
            diaa.append(min_distance)
        daa = np.mean(np.array(diaa))
        min_objectives_population = []
        min_objectives_true_front = []
        for i in range(len(true_front[0])):
            min_index_population = np.argmin(population[:, i])
            min_index_true_front = np.argmin(true_front[:, i])
            min_objectives_population.append(population[min_index_population])
            min_objectives_true_front.append(true_front[min_index_true_front])
        distance_sum = 0
        for point1, point2 in zip(min_objectives_population, min_objectives_true_front):
            distance = self.distance_square(point1, point2)
            distance = np.sqrt(distance)
            distance_sum += distance
        spread = (distance_sum + np.sum(np.abs(np.array(diaa) - daa))) / (distance_sum + len(population) * daa)
        return spread

    def pickle_save(self, file_path, solutions):
        """将字典保存到文件中"""
        file_pkl = file_path + ".pkl"
        with open(file_pkl, "wb") as f:
            pickle.dump(solutions, f)

    def pickle_read(self, file_path):
        """ 从文件加载保存的字典"""
        file_pkl = file_path + ".pkl"
        with open(file_pkl, "rb") as f:
            loaded_solutions = pickle.load(f)
        return loaded_solutions

    def csv_read(self, file_path, column_name):
        """读取csv文件的某列数据并存入列表"""
        data_frame = pd.read_csv(file_path)
        column_data = data_frame[column_name].tolist()
        return column_data


class PlotCsvCurve():
    """收敛曲线图类"""
    def __init__(self, csv_file, x_column, y_column):
        self.csv_file = csv_file
        self.x_column = x_column
        self.y_column = y_column

    def plot_csv_data(self):
        x_data = []
        y_data = []

        with open(self.csv_file, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # 跳过表头
            x_index = header.index(self.x_column)
            y_index = header.index(self.y_column)

            for row in reader:
                x_data.append(float(row[x_index]))
                y_data.append(float(row[y_index]))
        return x_data, y_data


class MyError(Exception):
    """异常类"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class FigGan():
    """画甘特图类"""
    def __init__(self, object):
        self.object = object  # 环境对象
        self.machine_dict = self.object.machine_dict  # 工件类型字典
        self.machine_tuple = self.object.machine_tuple  # 机器元组
        self.file_name = self.object.file_name  # 文件名
        self.order_count = self.object.order_count  # 新订单数
        self.C = self.object.completion_time
        self.T = self.object.delay_time_sum
        self.E = self.object.energy_consumption

    def figure(self):
        """画甘特图"""
        width = 12
        fig, ax = plt.subplots(figsize=(width, 6))  # Create a subplot with more vertical space
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Professional color palette
        kinds = ['K38A', 'K38B', 'K50']
        machines = ['M' + str(m + 1) for m in self.machine_tuple]

        # Plot tasks
        for machine, machine_object in self.machine_dict.items():
            for task_object in machine_object.task_list:
                kind = task_object.kind
                bar_width = task_object.time_end - task_object.time_begin
                bar_mid = task_object.time_begin + bar_width / 2
                ax.barh(machines[machine], bar_width,
                        left=task_object.time_begin, height=0.4,  # Less height for a slimmer bar
                        align='center', color=colors[kind % len(colors)],  # Cycle through colors
                        edgecolor='black', lw=0.5)  # Grey edge for a softer look
                # Position task label above the bar to avoid overlap and increase readability
                # ax.text(bar_mid, machines[machine], f'({task_object.number}-{task_object.task})',
                #         ha='center', va='bottom', color='black', fontsize=3)

        # Create the legend with hatch pattern for 'Machine breakdown'
        legend_patches = [mpatches.Patch(color=colors[i], label=kinds[i]) for i in range(len(kinds))]
        ax.legend(handles=legend_patches, loc='best', fontsize='medium', handlelength=2, handletextpad=1)

        # Customize the axis ticks for a cleaner look
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)

        # Save the Gantt chart as a vector graphic
        plt.savefig(self.file_name + '_' + str(self.C) + '_' + str(self.T) + '_' + str(self.E) + '_gantt.svg', bbox_inches='tight')
        # Optionally, show the plot if needed
        # plt.show()

    def figure_gpt(self):
        """Draw Gantt chart with an improved style for scientific papers."""
        width = 12
        fig, ax = plt.subplots(figsize=(width, 6))  # Create a subplot with more vertical space
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Professional color palette
        kinds = ['K38A', 'K38B', 'K50', 'Machine breakdown']
        machines = ['M' + str(m + 1) for m in self.machine_tuple]
        breakdown_hatch = '//'  # Hatch pattern for breakdowns

        # Plot tasks
        for machine, machine_object in self.machine_dict.items():
            for task_object in machine_object.task_list:
                kind = task_object.kind
                bar_width = task_object.time_end - task_object.time_begin
                bar_mid = task_object.time_begin + bar_width / 2
                ax.barh(machines[machine], bar_width,
                        left=task_object.time_begin, height=0.4,  # Less height for a slimmer bar
                        align='center', color=colors[kind % len(colors)],  # Cycle through colors
                        edgecolor='black', lw=0.5)  # Grey edge for a softer look
                # Position task label above the bar to avoid overlap and increase readability
                # ax.text(bar_mid, machines[machine], f'({task_object.number}-{task_object.task})',
                #         ha='center', va='bottom', color='black', fontsize=3)

        # Plot breakdown times
        for machine in self.machine_tuple:
            breakdowns = self.object.breakdown_m_dict.get(machine, [])
            for breakdown_start, breakdown_end in breakdowns:
                if breakdown_start < self.object.machine_dict[machine].time_end:
                    ax.barh(machines[machine], breakdown_end - breakdown_start,
                            left=breakdown_start, height=0.4,
                            align='center', color='grey', hatch='//',  # Hatch pattern for breakdowns
                            edgecolor='black', lw=0.5)

        # Create the legend with hatch pattern for 'Machine breakdown'
        legend_patches = [mpatches.Patch(color=colors[i], label=kinds[i]) for i in range(len(kinds) - 1)]
        legend_patches.append(mpatches.Patch(facecolor='grey', label=kinds[-1], hatch=breakdown_hatch))
        ax.legend(handles=legend_patches, loc='best', fontsize='medium', handlelength=2, handletextpad=1)

        # Customize the axis ticks for a cleaner look
        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)

        # Save the Gantt chart as a vector graphic
        plt.savefig(self.file_name + '_' + str(self.C) + '_' + str(self.T) + '_' + str(self.E) + '_gantt.svg', bbox_inches='tight')
        # Optionally, show the plot if needed
        # plt.show()


class AddData():
    """添加训练数据"""
    def __init__(self, path_file_name):
        self.file_name = path_file_name

    def add_data(self, data):
        with open(self.file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)


class Data():
    """读取csv文件数据类"""
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


# 测试各类
if __name__ == '__main__':
    # 添加训练数据
    # data_training = [1, 2, 3]
    # path_file_name = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/results/DA3C/training.csv'
    # result_data = AddData(path_file_name)
    # result_data.add_data(data_training)
    data_process = DataProcess()
    population = np.array([[283, 6315], [276, 6621], [267, 6373], [282, 6295], [301, 6638], [276, 6621], [304, 6850]])
    print(data_process.filter_pareto_front(population))

