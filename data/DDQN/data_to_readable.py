import os
import pandas as pd

# 定于读取文件夹
file_name_list = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']
file_name_write = ['P11', 'P21', 'P31', 'P41', 'P51', 'P61', 'P71', 'P81']

for file_name, file_write in zip(file_name_list,file_name_write):
    # 读取P.csv文件
    df = pd.read_csv('D:/Python project/Deep_Reinforcement_Learning_FJSP/data/DDQN/benchmark_batches/{}.csv'.format(file_name))

    # 重映射机器编号使其从0开始
    machine_mapping = {machine: index for index, machine in enumerate(sorted(df['machine'].unique()))}
    df['machine'] = df['machine'].map(machine_mapping)

    # 根据P.csv的数据得出kind_count和machine_count
    kind_count = df['lot'].nunique()
    machine_count = len(machine_mapping)  # 使用映射后的机器数量

    # 处理数据并创建process_data DataFrame
    processed_data = pd.DataFrame(columns=['kind', 'task', 'machine_selectable', 'process_time'])
    for (k, t), group in df.groupby(['lot', 'operation']):
        machines = tuple(group['machine'])
        times = tuple(group['proc-time'])
        processed_data = processed_data.append({
            'kind': k - 1,
            'task': t - 1,
            'machine_selectable': str(machines),
            'process_time': str(times)
        }, ignore_index=True)

    # 定义输出文件夹路径
    output_folder = 'D:/Python project/Deep_Reinforcement_Learning_FJSP/data/DDQN/{}'.format(file_write)

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 保存process_data.csv文件
    processed_data.to_csv(os.path.join(output_folder, 'process_data.csv'), index=False)

    # 创建并保存order_data.csv文件
    order_data = pd.DataFrame({
        'order': [0],
        'time_arrive': [0],
        'time_delivery': [1],
        'kind_number': [str(tuple(range(kind_count)))]  # 这里假设是示例数据---------------------------------------------------------
    })
    order_data.to_csv(os.path.join(output_folder, 'order_data.csv'), index=False)

    # 创建并保存based_data.csv文件，其中kind_count和machine_count是由P.csv计算得出
    based_data = pd.DataFrame({
        'kind_count': [kind_count],
        'machine_count': [machine_count],
        'order_count': [1],  # 这里假设只有一个订单
        'DDT': [1.0]        # 这里假设DDT是示例数据
    })
    based_data.to_csv(os.path.join(output_folder, 'based_data.csv'), index=False)

    # 输出提示信息
    print(f"Files have been created and saved in: {output_folder}")

