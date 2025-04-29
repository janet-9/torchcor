import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from models.dataset import Dataset
from models.cnn import CNN2d
from models.fno import FNO2d
from models.don import DeepONet2d
from models.wno import WNO2d
import argparse
from pathlib import Path
from tools import set_random_seed
import numpy as np

parser = argparse.ArgumentParser(description="root",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-root", type=str, default="/data/Bei")
args = parser.parse_args()

set_random_seed(42)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_id = torch.cuda.current_device()
gpu_name = torch.cuda.get_device_name(device_id)
print(gpu_name, flush=True)

dataset = Dataset(n_uac_points=100, root=args.root)
X_train = torch.tensor(dataset.X_train[:, :1, :, :])
x_min = X_train.min()
x_max = X_train.max()
X_train = (X_train - x_min) / (x_max - x_min)

X_test = torch.tensor(dataset.X_test[:, :1, :, :])
X_test = (X_test - x_min) / (x_max - x_min)

X_50 = torch.tensor(dataset.X_50[:, :1, :, :])
X_50 = (X_50 - x_min) / (x_max - x_min)

X_400 = torch.tensor(dataset.X_400[:, :1, :, :])
X_400 = (X_400 - x_min) / (x_max - x_min)

y_train = torch.tensor(dataset.y_train) - 1
y_test = torch.tensor(dataset.y_test) - 1
y_50 = torch.tensor(dataset.y_50) - 1
y_400 = torch.tensor(dataset.y_400) - 1

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
dataset_50 = TensorDataset(X_50, y_50)
dataset_400 = TensorDataset(X_400, y_400)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
loader_50 = DataLoader(dataset_50, batch_size=32)
loader_400 = DataLoader(dataset_400, batch_size=8)

error_100 = 99

# model = CNN2d().to(device)
# model = DeepONet2d().to(device)

for wavelet in ["haar", "db6", "coif3"]:
    for width in [32, 64, 128]:
        model = WNO2d(width=width, level=3, layers=2, size=[100, 100], wavelet=wavelet, in_channel=3, grid_range=[0, 1]).to(device)


# for modes1, modes2 in [(8, 8), (16, 16), (26, 26)]:
#     for width in [32, 64, 128]:
#         model = FNO2d(modes1=modes1, modes2=modes2, width=width, in_channels=1, out_dim=2, depth=4).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        error_list_100 = []
        error_list_50 = []
        error_list_400 = []
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_max_diff = 0
            train_abs_sum = 0
            total_train_samples = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            # Evaluation
            model.eval()
            test_abs_100 = []
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)

                    test_abs = torch.abs(outputs - targets)
                    test_abs_100.extend((test_abs.sum(dim=1) / 2).tolist())
            
            test_abs_50 = []
            with torch.no_grad():
                for inputs, targets in loader_50:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    test_abs = torch.abs(outputs - targets)
                    test_abs_50.extend((test_abs.sum(dim=1) / 2).tolist())

            test_abs_400 = []
            with torch.no_grad():
                for inputs, targets in loader_400:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    test_abs = torch.abs(outputs - targets)
                    test_abs_400.extend((test_abs.sum(dim=1) / 2).tolist())

            model_path = Path("./trained")
            model_path.mkdir(exist_ok=True, parents=True)
            test_error = sum(test_abs_100)/len(test_abs_100)
            if error_100 > test_error:
                torch.save(model, model_path / f"{model.name}.pth")
                error_100 = test_error
                error_list_100 = test_abs_100
                error_list_50 = test_abs_50
                error_list_400 = test_abs_400

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        # WNO:
        print((wavelet, width, width),
              round(np.array(error_list_50).mean(), 3), round(np.array(error_list_50).std(), 3),
              round(np.array(error_list_100).mean(), 3), round(np.array(error_list_100).std(), 3),
              round(np.array(error_list_400).mean(), 3), round(np.array(error_list_400).std(), 3),
              total_params)

        # FNO
        # print((modes1, modes2, width),
        #       round(np.array(error_list_50).mean(), 3), round(np.array(error_list_50).std(), 3),
        #       round(np.array(error_list_100).mean(), 3), round(np.array(error_list_100).std(), 3),
        #       round(np.array(error_list_400).mean(), 3), round(np.array(error_list_400).std(), 3),
        #       total_params)
