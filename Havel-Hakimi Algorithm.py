import numpy as np

def havel_hakimi(degree_sequence):
    n = len(degree_sequence)
    # 初始化邻接矩阵
    adj_matrix = np.zeros((n, n), dtype=int)
    
    # 创建一个列表来保存度序列与原始节点索引的对应关系
    indexed_degree_sequence = [(degree, i) for i, degree in enumerate(degree_sequence)]
    
    while True:
        # 排除度为0的节点
        indexed_degree_sequence = [x for x in indexed_degree_sequence if x[0] > 0]
        
        if len(indexed_degree_sequence) == 0:
            break
        
        # 按度排序，降序排列
        indexed_degree_sequence.sort(reverse=True, key=lambda x: x[0])
        
        # 取出最大度和对应的节点
        d, node = indexed_degree_sequence.pop(0)
        
        if d > len(indexed_degree_sequence):
            raise ValueError("The given degree sequence is not graphical.")
        
        # 与其他度为正的节点连接
        for i in range(d):
            neighbor_degree, neighbor_node = indexed_degree_sequence[i]
            indexed_degree_sequence[i] = (neighbor_degree - 1, neighbor_node)
            adj_matrix[node][neighbor_node] = 1
            adj_matrix[neighbor_node][node] = 1
    
    return adj_matrix

# 默认的度序列
degree_sequence = [6, 4, 3, 3, 2, 1, 1]

# 得到邻接矩阵
adj_matrix = havel_hakimi(degree_sequence)

# 输出结果，按行打印并格式化输出
output = "["
for i, row in enumerate(adj_matrix):
    if i < len(adj_matrix) - 1:  # 如果不是最后一行
        output += f"[{', '.join(map(str, row))}],\n"
    else:  # 最后一行
        output += f"[{', '.join(map(str, row))}]"
output += "]"
print(output)
