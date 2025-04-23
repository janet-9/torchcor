import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pathlib import Path
import numpy as np
from torchcor.core.mesh import MeshReader
from sklearn.model_selection import train_test_split
from tools import Normalizer
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATConv, GCNConv, GINConv, SAGEConv, NNConv


class ConductivityGAT(nn.Module):
    def __init__(self, hidden_channels=64, dropout=0.2):
        super(ConductivityGAT, self).__init__()

        self.dropout = dropout

        self.branch_conv1 = GATConv(2, hidden_channels, aggr='mean') 
        self.branch_bn1 = BatchNorm(hidden_channels)
        self.branch_conv2 = GATConv(hidden_channels, hidden_channels, aggr='mean')
        self.branch_bn2 = BatchNorm(hidden_channels)
        self.branch_lin = nn.Linear(hidden_channels, hidden_channels)


        self.trunk_conv1 = GATConv(2, hidden_channels, aggr='mean') 
        self.trunk_bn1 = BatchNorm(hidden_channels)
        self.trunk_conv2 = GATConv(hidden_channels, hidden_channels, aggr='mean')
        self.trunk_bn2 = BatchNorm(hidden_channels)
        self.trunk_lin = nn.Linear(hidden_channels, hidden_channels)

        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # x = x + x1

        # Second layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # x = x + x2
        # Fully connected layers
        x = self.lin1(x)
        x = F.relu(x)

        return x


class EdgeAwareGNN(nn.Module):
    def __init__(self, in_channels=1, edge_dim=3, hidden_channels=16, dropout=0.2):
        super().__init__()

        # MLP to generate dynamic weights from edge features
        self.nn1 = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = NNConv(in_channels, hidden_channels, self.nn1, aggr='mean')
        self.bn1 = BatchNorm(hidden_channels)

        # self.nn2 = nn.Sequential(
        #     nn.Linear(edge_dim, hidden_channels),
        #     nn.ReLU(),
        #     nn.Linear(hidden_channels, hidden_channels * hidden_channels)
        # )
        # self.conv2 = NNConv(hidden_channels, hidden_channels, self.nn2, aggr='mean')
        # self.bn2 = BatchNorm(hidden_channels)

        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = nn.Linear(hidden_channels // 2, 1)

        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        for conv, bn in [(self.conv1, self.bn1)]: 
                        #  (self.conv2, self.bn2)]:
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.sigmoid(self.lin1(x))
        return self.lin2(x)



def load_stimulus_region(vtx_filepath):
    with Path(vtx_filepath).open("r") as f:
        region_size = int(f.readline().strip())

    region = np.loadtxt(vtx_filepath, dtype=int, skiprows=2)
    
    if len(region) != region_size:
        raise Exception(f"Error loading {vtx_filepath}")
    
    return torch.from_numpy(region)

at_min = 500
at_max = 0
solution_dir = Path("/data/Bei/atrium_at_rt")
for case_dir in solution_dir.iterdir():
    case_infos = case_dir.name.split("_")
    case_name = case_infos[0] + "_" + case_infos[1]
    ATs = torch.load(case_dir / "ATs.pt", weights_only=True)
    at_min = at_min if ATs.min() > at_min else ATs.min()
    at_max = at_max if ATs.max() < at_max else ATs.max()

normalizer = Normalizer()
normalizer.y_min = at_min
normalizer.y_max = at_max
print(at_min, at_max)

dataset_file = Path("/data/Bei/dataset.pt")
if dataset_file.exists():
    dataset = torch.load(dataset_file, weights_only=False)
else:
    dataset = []
    solution_dir = Path("/data/Bei/atrium_at_rt")
    data_dir = Path("/data/Bei/meshes_refined")
    for case_dir in solution_dir.iterdir():
        case_infos = case_dir.name.split("_")
        case_name = case_infos[0] + "_" + case_infos[1]

        reader = MeshReader(data_dir / case_name)
        nodes, _, _, _ = reader.read()

        stimulus_path = (data_dir / case_name / case_dir.name).with_suffix(".vtx")
        stimulus_location = load_stimulus_region(stimulus_path)
        stimulus_nodes = nodes[stimulus_location]
        dists = torch.cdist(torch.from_numpy(nodes), torch.from_numpy(stimulus_nodes), p=2).min(dim=1)[0].unsqueeze(dim=1)
        min_dist = dists.min()
        max_dist = dists.max()
        dists = (dists - min_dist) / (max_dist - min_dist + 1e-8)

        ATs = torch.load(case_dir / "ATs.pt", weights_only=True)
        X = torch.zeros_like(ATs)
        X[stimulus_location] = 1
        X = X.view(-1, 1)
        uac_coordinate = torch.from_numpy(np.load(data_dir / case_name / "UAC.npy"))
        X = torch.cat([X, uac_coordinate, dists], dim=1)

        M = torch.load(case_dir / "M.pt", weights_only=True)
        # K = torch.load(case_dir / "K.pt", weights_only=True)
        # A= torch.load(case_dir / "A.pt", weights_only=True)

        # mka = torch.stack([M.values(), K.values(), A.values()], dim=1)
        # data = Data(x=X.view(-1, 1), edge_index=M.indices(), edge_attr=mka, y=ATs.view(-1, 1))
        y = normalizer.normalize(ATs.view(-1, 1))
        data = Data(x=X.float(), edge_index=M.indices(), y=y.float())
        dataset.append(data)
    
    torch.save(dataset, "/data/Bei/dataset.pt")

train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)
train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = ScalarGCN()
# model = EdgeAwareGNN()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

loss_fn = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()

def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        # print(output.mean().item(), output.median().item(), output.max().item(), output.min().item())
        loss = loss_fn(output, batch.y) # + 0.5 * loss_fn(output, batch.x[:, -1].unsqueeze(1))

        # stimulus_mask = batch.x[:, -1]
        # stimulus_loss = torch.mean((output[stimulus_mask] - batch.y[stimulus_mask]) ** 2) / stimulus_mask.sum()
        # loss = loss_fn(output, batch.y) 
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()

    return total_loss / len(train_loader)

def test():
    model.eval()
    total_error = 0
    max_error = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            error = torch.abs(normalizer.inverse_normalize(output) - normalizer.inverse_normalize(batch.y))

            total_error += error.mean().item()
            m_error = error.max().item()
            max_error = m_error if max_error < m_error else max_error
    return total_error / len(test_loader), max_error


for _ in range(100):
    train_loss = train()
    test_error = test()
    print(train_loss, test_error)
