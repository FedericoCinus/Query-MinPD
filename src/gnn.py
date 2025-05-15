import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def propagate_with_gcn(G, sampled_indices, real_values, hidden_channels=16, epochs=200, lr=0.01):
    N = G.number_of_nodes()  # Total number of nodes
    sampled_values = real_values[sampled_indices]

    # Node features (X)
    X = torch.zeros((N, 1), dtype=torch.float32)
    X[sampled_indices] = torch.tensor(sampled_values, dtype=torch.float32)

    # Convert NetworkX graph to edge_index
    data = from_networkx(G)
    edge_index = data.edge_index

    # Known labels (y)
    y = torch.zeros((N, 1), dtype=torch.float32)
    y[sampled_indices] = torch.tensor(sampled_values, dtype=torch.float32)

    # Train mask (train_mask)
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[sampled_indices] = True

    # Create data object
    data = Data(x=X, edge_index=edge_index, y=y, train_mask=train_mask)

    # Define and train the GCN model
    model = GCN(num_node_features=1, hidden_channels=hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    for _epoch in range(epochs):
        train()

    # Predict values for all nodes
    model.eval()
    predicted_values = model(data.x, data.edge_index).detach().numpy()

    return predicted_values, sampled_indices, 0.


# class GCN(torch.nn.Module): # COMPUATIONAL HEAVY AND NOT EFFECTIVE
#     def __init__(self, num_node_features, hidden_channels):
#         super(GCN, self).__init__()
#         self.conv1 = GCNConv(num_node_features, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, 1)

#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         return x

# def get_structural_features(G, real_values, sampled_indices):
#     # Node degree
#     degree = np.array([G.degree(n) for n in G.nodes()]).reshape(-1, 1)
    
#     # Clustering coefficient
#     clustering_coeff = np.array([nx.clustering(G, n) for n in G.nodes()]).reshape(-1, 1)
    
#     # PageRank
#     pagerank = np.array([nx.pagerank(G)[n] for n in G.nodes()]).reshape(-1, 1)
    
#     # Betweenness centrality
#     betweenness_centrality = np.array([nx.betweenness_centrality(G)[n] for n in G.nodes()]).reshape(-1, 1)
    
#     # Combine all features
#     combined_features = np.hstack([degree, clustering_coeff, pagerank, betweenness_centrality])
    
#     # Add real_values to combined_features for sampled nodes
#     combined_features[sampled_indices, 0] = real_values[sampled_indices].flatten()
    
#     return combined_features

# def propagate_with_gcn(G, sampled_indices, real_values, hidden_channels=16, epochs=200, lr=0.01):
#     N = G.number_of_nodes()  # Total number of nodes
#     sampled_values = real_values[sampled_indices]

#     # Get structural features
#     combined_features = get_structural_features(G, real_values, sampled_indices)
    
#     # Convert combined features to tensor
#     X = torch.tensor(combined_features, dtype=torch.float32)

#     # Convert NetworkX graph to edge_index
#     data = from_networkx(G)
#     edge_index = data.edge_index

#     # Known labels (y)
#     y = torch.zeros((N, 1), dtype=torch.float32)
#     y[sampled_indices] = torch.tensor(sampled_values, dtype=torch.float32)

#     # Train mask (train_mask)
#     train_mask = torch.zeros(N, dtype=torch.bool)
#     train_mask[sampled_indices] = True

#     # Create data object
#     data = Data(x=X, edge_index=edge_index, y=y, train_mask=train_mask)

#     # Define and train the GCN model
#     model = GCN(num_node_features=combined_features.shape[1], hidden_channels=hidden_channels)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     def train():
#         model.train()
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index)
#         loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
#         loss.backward()
#         optimizer.step()

#     for epoch in range(epochs):
#         train()

#     # Predict values for all nodes
#     model.eval()
#     predicted_values = model(data.x, data.edge_index).detach().numpy()

#     return predicted_values, sampled_indices, 0.