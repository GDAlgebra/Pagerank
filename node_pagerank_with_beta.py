import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

# 参数设置
def compute_pagerank(adj_matrix, beta, eta):
    """计算PageRank中心性"""
    n = len(adj_matrix)
    R = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
    R = np.nan_to_num(R)  # 处理分母为0的情况
    identity_matrix = np.eye(n)
    return np.linalg.inv(identity_matrix - beta * R.T) @ ((1 - beta) * eta)

# 根据策略构造邻接矩阵
def build_adjacency_matrix(strategy, n, initial_adj_matrix):
    """根据策略构建邻接矩阵，仅修改指定节点的行"""
    adj_matrix = initial_adj_matrix.copy()
    for i, links in enumerate(strategy):
        adj_matrix[i, :] = 0  # 清空该节点的所有出边
        for j in links:
            adj_matrix[i, j] = 1  # 添加新策略的出边
    return adj_matrix

# 计算所有顶点的PageRank与beta的关系
def compute_all_nodes_pagerank_vs_beta():
    n = 6  # 节点数
    eta = np.array([0, 1, 0, 0, 0, 0])  
    beta_values = np.linspace(0.01, 0.99, 50)  # beta在(0, 1)之间的取值

    # 初始邻接矩阵
    initial_adj_matrix = np.array([
        [0, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0]
    ])

    # 计算每个节点的出度
    di_list = np.sum(initial_adj_matrix, axis=1).astype(int)

    # 初始化策略空间
    strategies = {i: list(combinations([j for j in range(n) if j != i], di_list[i])) for i in range(n)}  # 每个节点选择d_i个邻居
    results = {node: {response: [] for response in strategies[node]} for node in range(n)}

    for beta in beta_values:
        for node in range(n):
            for response in strategies[node]:
                # 构建策略
                current_strategy = [[j for j in range(n) if initial_adj_matrix[i][j] == 1] if i != node else list(response) for i in range(n)]
                adj_matrix = build_adjacency_matrix(current_strategy, n, initial_adj_matrix)
                pagerank = compute_pagerank(adj_matrix, beta, eta)
                results[node][response].append(pagerank[node])  # 记录当前节点的PageRank

    # 绘图
    for node in range(n):
        plt.figure(figsize=(8, 6))
        for response, pageranks in results[node].items():
            plt.plot(beta_values, pageranks, label=f"Response {response}")
        plt.xlabel("Beta")
        plt.ylabel(f"PageRank of Node {node}")
        plt.title(f"PageRank of Node {node} vs Beta for Different Responses")
        plt.legend()
        plt.grid()
        plt.show()

# 主要功能
def main():
    print("Plotting PageRank of all nodes vs Beta for all responses...")
    compute_all_nodes_pagerank_vs_beta()

if __name__ == "__main__":
    main()
