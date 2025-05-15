import numpy as np

def random_reconstruction(G, sampled_indices, real_values):

    # Get known values for sampled nodes
    values = []
    for idx in sampled_indices:
        values.append(real_values[idx].item())
    
    # Convert node_values to numpy array
    node_values_array = np.random.uniform(min(values), max(values), G.number_of_nodes()).reshape(-1, 1)
    
    return node_values_array, sampled_indices, 0.