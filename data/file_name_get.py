"""生成算例名"""
DDT_list = [0.5, 1.0, 1.5]
machine_count_list = [10, 15, 20]
order_count_list = [5, 10, 15]
file_name_list = []
# 生成各实例文件
for DDT in DDT_list:
    for M in machine_count_list:
        for S in order_count_list:
            file_name = 'DDT' + str(DDT) + '_M' + str(M) + '_R' + str(S)  # 算例文件夹名
            file_name_list.append(file_name)
print(file_name_list)

