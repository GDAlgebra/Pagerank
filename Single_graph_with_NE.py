import numpy as np
import itertools

def compute_pagerank(adj_matrix, beta, eta):
    """Calculate PageRank centrality"""
    n = len(adj_matrix)
    R = np.divide(
        adj_matrix,
        adj_matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(adj_matrix, dtype=float),
        where=adj_matrix.sum(axis=1, keepdims=True) != 0
    )
    identity_matrix = np.eye(n)
    return np.linalg.inv(identity_matrix - beta * R.T) @ ((1 - beta) * eta)

def find_better_choices(adj_matrix, beta, eta):
    """Find vertices with better choices and their new configurations"""
    n = len(adj_matrix)
    current_pagerank = compute_pagerank(adj_matrix, beta, eta)
    better_choices = []

    for i in range(n):
        # Store the original row
        original_row = adj_matrix[i].copy()
        out_degree = int(original_row.sum())

        # Skip if no outgoing edges
        if out_degree == 0:
            continue

        # Iterate over all possible configurations of outgoing edges
        for indices in itertools.combinations([j for j in range(n) if j != i], out_degree):
            new_row = np.zeros(n)
            new_row[list(indices)] = 1

            # Modify the adjacency matrix for the current configuration
            adj_matrix[i] = new_row

            # Compute the new PageRank values
            new_pagerank = compute_pagerank(adj_matrix, beta, eta)

            # Restore the original row
            adj_matrix[i] = original_row

            # Check if the PageRank of the current node increases
            if new_pagerank[i] > current_pagerank[i]:
                better_choices.append((i, list(indices), new_pagerank[i]))

    return better_choices

# Example input
adj_matrix = np.array(
[[0, 1, 1, 1, 1, 1, 1],
[1, 0, 1, 1, 1, 0, 0],
[1, 1, 0, 1, 0, 0, 0],
[1, 1, 1, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 0],
[1, 0, 0, 0, 0, 0, 0]], dtype=float)

beta = 0.95
eta = np.ones(len(adj_matrix)) / len(adj_matrix)

# Find vertices with better choices
better_choices = find_better_choices(adj_matrix, beta, eta)

if better_choices:
    print("Better choices found:")
    for vertex, new_edges, new_pagerank in better_choices:
        print(f"Vertex {vertex} can improve with edges {new_edges}, new PageRank: {new_pagerank}")
else:
    print("Graph is in Nash Equilibrium")
