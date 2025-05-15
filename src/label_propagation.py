import networkx as nx
import numpy as np

def propagate_with_label_prop(G, sampled_indices, real_values, max_iter=200):
    # Initialize node values with zeros
    node_values = {node: 0 for node in G.nodes()}
    
    # Set known values for sampled nodes
    for idx in sampled_indices:
        node_values[idx] = real_values[idx].item()

    sampled_nodes = set(sampled_indices)

    # Label Propagation Algorithm
    for _ in range(max_iter):
        new_values = node_values.copy()
        for node in G.nodes():
            if node in sampled_nodes:
                continue
            neighbors = list(G.neighbors(node))
            if neighbors:  # Avoid division by zero
                new_values[node] = np.mean([node_values[neighbor] for neighbor in neighbors])
        node_values = new_values
    
    # Convert node_values to numpy array
    node_values_array = np.array([node_values[node] for node in G.nodes()]).reshape(-1, 1)
    
    return node_values_array, sampled_indices, 0.