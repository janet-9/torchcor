import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class BranchTrunkNet(nn.Module):
    def __init__(self, in_channels=2, hidden_channels=64, out_channels=64, heads=2, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, batch):
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)
        output = self.mlp(x)    # [batch_size, out_channels]
        return output
    

class DeepONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_net = BranchTrunkNet(in_channels=2, hidden_channels=64, out_channels=64, heads=2, dropout=0.2)
        self.trunk_net = BranchTrunkNet(in_channels=2, hidden_channels=64, out_channels=64, heads=2, dropout=0.2)

        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def forward(self, data):
        branch_out = self.brank_net(data.x[0], data.edge_index, data.batch)
        trunk_out = self.trunk_net(data.x[1], data.edge_index, data.batch)

        combined = branch_out * trunk_out

        output = self.mlp(combined)
        return output
    























