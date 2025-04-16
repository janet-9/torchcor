import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from models import ScalarGCN


def load_stimulus_region(vtx_filepath):
    with Path(vtx_filepath).open("r") as f:
        region_size = int(f.readline().strip())

    region = np.loadtxt(vtx_filepath, dtype=int, skiprows=2)
    
    if len(region) != region_size:
        raise Exception(f"Error loading {vtx_filepath}")
    
    return torch.from_numpy(region)

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

        stimulus_path = (data_dir / case_name / case_dir.name).with_suffix(".vtx")
        stimulus_location = load_stimulus_region(stimulus_path)
        ATs = torch.load(case_dir / "ATs.pt", weights_only=True)
        X = torch.zeros_like(ATs)
        X[stimulus_location] = 1
        connectivity = torch.load(case_dir / "M.pt", weights_only=True).indices()
        data = Data(x=X.view(-1, 1), edge_index=connectivity, y=ATs.view(-1, 1))
        dataset.append(data)
    
    torch.save(dataset, "/data/Bei/dataset.pt")

train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)

train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
test_loader = DataLoader(test_data, batch_size=2, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ScalarGCN()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()

def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def test():
    model.eval()
    total_error = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            error = criterion(output, batch.y)
            total_error += error.item()
    return total_error / len(test_loader)


for _ in range(100):
    train_loss = train()
    test_error = test()
    print(train_loss, test_error)
