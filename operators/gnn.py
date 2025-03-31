import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MessagePassing

class GNNWithEdgeFeatures(MessagePassing):
    def __init__(self, in_channels, edge_dim, out_channels):
        super(GNNWithEdgeFeatures, self).__init__(aggr='mean')
        self.node_lin = torch.nn.Linear(in_channels, out_channels)
        self.edge_lin = torch.nn.Linear(edge_dim, out_channels)
        self.out_lin = torch.nn.Linear(out_channels, 1)  # Single feature value

    def forward(self, x, edge_index, edge_attr):
        x = self.node_lin(x)
        edge_attr = self.edge_lin(edge_attr)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.out_lin(F.relu(aggr_out))

# Example usage
num_nodes, num_edges = 10, 20
node_features = torch.rand((num_nodes, 16))  # Node features
edge_features = torch.rand((num_edges, 8))  # Edge features
edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Edge connections

model = GNNWithEdgeFeatures(16, 8, 32)
output = model(node_features, edge_index, edge_features)
print(output.shape)  # [num_nodes, 1]
