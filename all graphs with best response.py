import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations, combinations

# 根据出度序列生成所有可能的有向图邻接矩阵
def generate_graphs_from_outdegree_sequence(outdegree_sequence):
    """根据出度序列生成所有可能的有向图的邻接矩阵"""
    n = len(outdegree_sequence)
    possible_graphs = []

    def is_valid_adjacency_matrix(matrix, outdegree_sequence):
        return list(np.sum(matrix, axis=1)) == outdegree_sequence

    def recursive_graph_build(matrix, degrees, idx):
        if idx >= n:
            if is_valid_adjacency_matrix(matrix, outdegree_sequence):
                possible_graphs.append(matrix.copy())
            return

        # 当前节点度数为0时，直接跳过到下一个节点
        if degrees[idx] == 0:
            recursive_graph_build(matrix, degrees, idx + 1)
            return

        # 尝试为当前节点连接其他节点
        for neighbors in combinations(range(n), degrees[idx]):
            matrix[idx, :] = 0  # 清空当前节点的所有出边
            for neighbor in neighbors:
                if idx != neighbor:
                    matrix[idx, neighbor] = 1

            recursive_graph_build(matrix, degrees, idx + 1)

            # 回溯
            matrix[idx, :] = 0

    # 初始化递归生成
    initial_matrix = np.zeros((n, n), dtype=int)
    recursive_graph_build(initial_matrix, outdegree_sequence[:], 0)

    return possible_graphs

# 移除同构图
def remove_isomorphic_graphs(graphs):
    """从生成的图中移除同构的图，只保留不同构的图"""
    unique_graphs = []
    seen_graphs = set()

    for graph in graphs:
        g = nx.DiGraph(graph)
        # 使用自定义方法生成哈希值以避免依赖scipy
        canonical_form = nx.to_numpy_array(g)
        canonical_hash = hash(canonical_form.tobytes())

        if canonical_hash not in seen_graphs:
            unique_graphs.append(graph)
            seen_graphs.add(canonical_hash)

    return unique_graphs

# 计算PageRank中心性
def compute_pagerank(adj_matrix, beta, eta):
    """计算PageRank中心性"""
    n = len(adj_matrix)
    R = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)
    R = np.nan_to_num(R)  # 处理分母为0的情况
    identity_matrix = np.eye(n)
    return np.linalg.inv(identity_matrix - beta * R.T) @ ((1 - beta) * eta)

def is_nash_equilibrium(adj_matrix, outdegree_sequence):
    """判断给定图是否是Nash均衡"""
    n = len(adj_matrix)
    eta = np.ones(n) / n  # 均匀分布的eta
    current_pagerank = compute_pagerank(adj_matrix, beta=0.85, eta=eta)

    for node in range(n):
        original_rank = np.floor(current_pagerank[node] * 1e10) / 1e10  # 去尾处理
        best_response_found = False

        for neighbors in combinations(range(n), outdegree_sequence[node]):
            if node in neighbors:
                continue

            # 创建一个新的邻接矩阵，模拟新的策略
            new_adj_matrix = adj_matrix.copy()
            new_adj_matrix[node, :] = 0  # 清空当前节点的所有出边
            for neighbor in neighbors:
                new_adj_matrix[node, neighbor] = 1

            # 计算新的PageRank
            new_pagerank = compute_pagerank(new_adj_matrix, beta=0.85, eta=eta)
            new_rank = np.floor(new_pagerank[node] * 1e10) / 1e10  # 去尾处理

            if new_rank > original_rank:
                best_response_found = True
                break

        if best_response_found:
            return "Not Nash Equilibrium"

    return "Nash Equilibrium"

# 绘制所有图到一张图中
def draw_combined_graphs(graphs, titles):
    """将多个图合并绘制到一张图中"""
    num_graphs = len(graphs)
    cols = 3  # 每行显示的图数量
    rows = (num_graphs + cols - 1) // cols  # 计算行数

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()

    for i, (graph, title) in enumerate(zip(graphs, titles)):
        g = nx.DiGraph(graph)
        pos = nx.spring_layout(g)  # 布局
        nx.draw(g, pos, ax=axes[i], with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10)
        axes[i].set_title(title)

    # 如果图数量不足，隐藏多余的子图
    for i in range(num_graphs, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# 主函数：给定出度序列，输出所有满足条件的有向图邻接矩阵
# 并判断每个图是否为Nash均衡
# 同时绘制所有Nash均衡对应的图
def main():
    outdegree_sequence = [1, 2, 2, 2, 2, 2]  # 示例出度序列
    print(f"Generating all valid directed adjacency matrices for outdegree sequence {outdegree_sequence}...")
    all_graphs = generate_graphs_from_outdegree_sequence(outdegree_sequence)
    unique_graphs = remove_isomorphic_graphs(all_graphs)

    print(f"Total number of unique graphs: {len(unique_graphs)}")
    nash_graphs = []
    nash_titles = []

    for i, graph in enumerate(unique_graphs):
        status = is_nash_equilibrium(graph, outdegree_sequence)
        if status == "Nash Equilibrium":
            print(f"Graph {i+1} adjacency matrix:")
            print(graph)
            nash_graphs.append(graph)
            nash_titles.append(f"Graph {i+1}: Nash Equilibrium")

    # 绘制所有Nash均衡的图到一张图中
    if nash_graphs:
        draw_combined_graphs(nash_graphs, nash_titles)

if __name__ == "__main__":
    main()
