import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt
from datetime import datetime

# 参数设置
n = 20  # 节点数量
di = 3  # 每个节点出边数量
beta = 0.85  # 折扣因子
eta = np.random.dirichlet(np.ones(n))  # 每个节点的固有中心性，随机概率分布

# 工具函数：计算PageRank中心性
def compute_pagerank(adj_matrix, beta, eta):
    n = len(adj_matrix)
    R = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
    R = np.nan_to_num(R)  # 处理分母为0的情况
    identity_matrix = np.eye(n)
    return np.linalg.inv(identity_matrix - beta * R.T) @ ((1 - beta) * eta)

# 初始化博弈
players = list(range(n))
strategies = [list(combinations([j for j in players if j != i], di)) for i in players]

# 随机初始化策略（每个节点选择di个出边）
def random_strategy():
    return [list(strategies[i][np.random.choice(len(strategies[i]))]) for i in range(n)]

# 根据策略构造邻接矩阵
def build_adjacency_matrix(strategy, n):
    adj_matrix = np.zeros((n, n))
    for i, links in enumerate(strategy):
        for j in links:
            adj_matrix[i, j] = 1
    return adj_matrix

# 检测强连通分支
def detect_strongly_connected_components(strategy, n):
    G = nx.DiGraph()
    for i in range(n):
        for j in strategy[i]:
            G.add_edge(i, j)
    scc = list(nx.strongly_connected_components(G))
    return scc

# 保存最终网络图结构
def save_final_graph(strategy, n, save_path):
    G = nx.DiGraph()
    for i in range(n):
        for j in strategy[i]:
            G.add_edge(i, j)

    # 获取强连通分支
    scc = list(nx.strongly_connected_components(G))
    pos = {}
    x_offset = 0
    y_offset = 0
    for component in scc:
        subgraph = G.subgraph(component)
        sub_pos = nx.spring_layout(subgraph, k=0.5, iterations=100)
        for node, coords in sub_pos.items():
            pos[node] = coords + np.array([x_offset, y_offset])
        x_offset += 3  # 增加水平偏移确保分支间距更远

    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray', arrowsize=20)
    plt.title("Final Network Graph")
    plt.savefig(save_path)
    plt.show()  # 显示图

# 效用函数：节点的PageRank值
def utility(strategy, player):
    adj_matrix = build_adjacency_matrix(strategy, n)
    pagerank = compute_pagerank(adj_matrix, beta, eta)
    return pagerank[player]

# 最优响应计算
def best_response(current_strategy, player):
    max_utility = -np.inf
    best_action = None
    for action in strategies[player]:
        new_strategy = current_strategy.copy()
        new_strategy[player] = action
        u = utility(new_strategy, player)
        if u > max_utility:
            max_utility = u
            best_action = action
    return best_action

# 最优响应计算，返回随机的最优响应，如果更新后的效用不高则返回 None
def best_responses(current_strategy, player):
    max_utility = utility(current_strategy, player)  # 初始化效用为当前策略的效用
    best_actions = []  # 用来存储所有的最优响应动作
    
    # 遍历玩家的所有策略
    for action in strategies[player]:
        new_strategy = current_strategy.copy()
        new_strategy[player] = action
        u = utility(new_strategy, player)
        
        if u > max_utility:
            max_utility = u
            best_actions = [action]  # 找到更大的效用时，更新最优响应集合
        elif u == max_utility:
            best_actions.append(action)  # 如果效用相等，添加该动作到最优响应集合

    # 如果没有找到更高效用的最优响应，则返回 None
    if not best_actions or max_utility <= utility(current_strategy, player):
        return None

    # 从最优响应集合中随机选择一个
    return random.choice(best_actions)



# 模拟博弈
strategy = random_strategy()  # 随机初始化策略
iteration = 0
all_iterations_output = []
while True:
    iteration += 1
    iteration_output = []
    iteration_output.append(f"Iteration {iteration}")
    updated_edges = []
    equilibrium = True
    for player in players:
        current_action = strategy[player]
        best_action = best_response(strategy, player)
        if best_action != current_action:
            strategy[player] = best_action
            equilibrium = False
            iteration_output.append(f"Player {player} updated strategy to {best_action}")
            # 标记更新的边
            updated_edges.extend([(player, neighbor) for neighbor in best_action if neighbor not in current_action])

    # 计算当前网络的PageRank分数
    adj_matrix = build_adjacency_matrix(strategy, n)
    pagerank = compute_pagerank(adj_matrix, beta, eta)
    iteration_output.append(f"Current PageRank: {pagerank}\n")
    all_iterations_output.append("\n".join(iteration_output))

    # 将迭代结果展示在终端
    print("\n".join(iteration_output))

    # 检查是否达到均衡
    if equilibrium:
        iteration_output.append("Equilibrium reached.")
        all_iterations_output.append("\n".join(iteration_output))
        print("Equilibrium reached.")
        break

# 最终结果
final_output = []
final_output.append("Final Strategy:")
final_output.append(str(strategy))
final_adj_matrix = build_adjacency_matrix(strategy, n)
final_pagerank = compute_pagerank(final_adj_matrix, beta, eta)
final_output.append("Final PageRank:")
final_output.append(str(final_pagerank))

# 检测强连通分支
scc = detect_strongly_connected_components(strategy, n)
final_output.append("Strongly Connected Components:")
final_output.append(str(scc))

# 保存最终网络图和策略数据
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
graph_filename = f"final_graph_{current_datetime}.png"
data_filename = f"game_data_{current_datetime}.csv"
iterations_filename = f"iterations_output_{current_datetime}.txt"

save_final_graph(strategy, n, save_path=graph_filename)
np.savetxt(data_filename, final_adj_matrix, delimiter=",", fmt="%d", header="Adjacency Matrix")

# 保存所有迭代的输出到文本文件
with open(iterations_filename, "w") as f:
    f.write("\n\n".join(all_iterations_output))
    f.write("\n\n".join(final_output))

print(f"Final graph saved as {graph_filename}")
print(f"Game data saved as {data_filename}")
print(f"Iterations output saved as {iterations_filename}")
print(f"Strongly Connected Components: {scc}")
