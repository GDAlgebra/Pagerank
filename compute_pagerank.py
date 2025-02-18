import numpy as np
import networkx as nx
from itertools import combinations
import copy
import matplotlib.pyplot as plt

# 参数设置
beta = 0.85  # 折扣因子
eta = np.array([0.6, 0.1, 0.1, 0.1])  # 使用用户提供的预设概率分布

# 工具函数：计算PageRank中心性
def compute_pagerank(adj_matrix, beta, eta):
    n = len(adj_matrix)
    R = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
    R = np.nan_to_num(R)  # 处理分母为0的情况
    identity_matrix = np.eye(n)
    return np.linalg.inv(identity_matrix - beta * R.T) @ ((1 - beta) * eta)

# 输入邻接矩阵并计算PageRank
def input_and_compute_pagerank(adj_matrix, beta, eta):
    pagerank = compute_pagerank(adj_matrix, beta, eta)
    return pagerank

# 根据策略构造邻接矩阵
def build_adjacency_matrix(strategy, n):
    adj_matrix = np.zeros((n, n))
    for i, links in enumerate(strategy):
        for j in links:
            adj_matrix[i, j] = 1
    return adj_matrix

# 计算所有响应及对应的PageRank
def all_responses_with_pagerank(current_strategy, player, strategies, eta):
    response_pageranks = []
    for action in strategies[player]:
        new_strategy = copy.deepcopy(current_strategy)
        new_strategy[player] = action
        adj_matrix = build_adjacency_matrix(new_strategy, len(new_strategy))
        pagerank = compute_pagerank(adj_matrix, beta, eta)
        response_pageranks.append((action, pagerank))
    return response_pageranks

# 绘制邻接矩阵对应的图结构
def plot_graph_from_adj_matrix(adj_matrix):
    G = nx.DiGraph(adj_matrix)
    pos = nx.spring_layout(G)  # 使用spring布局算法
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=12)
    plt.title("Graph Structure from Adjacency Matrix")
    plt.show()

# 主要功能：使用预设邻接矩阵，计算PageRank和所有节点的响应及PageRank
def main():
    # 预设的邻接矩阵，顶点出度分别为1，2，2，2
    n = 4  # 顶点数量
    adj_matrix = np.array([
        [0, 1, 0, 0],  # 顶点0出度为1
        [1, 0, 1, 0],  # 顶点1出度为2
        [0, 1, 0, 1],  # 顶点2出度为2
        [0, 1, 1, 0]   # 顶点3出度为2
    ])

    di_list = [sum(row) for row in adj_matrix]  # 根据邻接矩阵动态确定每个顶点的出度
    print("Using predefined adjacency matrix:")
    print(adj_matrix)
    print(f"Out-degree of each vertex: {di_list}")

    print("\nCalculating PageRank...")
    pagerank = input_and_compute_pagerank(adj_matrix, beta, eta)
    print(f"PageRank of each vertex: {pagerank}")

    print("\nCalculating All Responses and Corresponding PageRanks...")
    current_strategy = [[j for j in range(n) if adj_matrix[i][j] == 1] for i in range(n)]
    strategies = [list(combinations([j for j in range(n) if j != i], int(di_list[i]))) for i in range(n)]  # 排除自身
    for player in range(n):
        all_responses = all_responses_with_pagerank(current_strategy, player, strategies, eta)
        print(f"\nPlayer {player}: All Responses and Corresponding PageRanks")
        for action, pagerank in all_responses:
            print(f"  Response: {action}, PageRank: {pagerank}")

    # print("\nPlotting Graph Structure...")
    # plot_graph_from_adj_matrix(adj_matrix)

if __name__ == "__main__":
    main()
