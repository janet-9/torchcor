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

X_extra = torch.tensor(dataset.X_400[:, :1, :, :])
X_extra = (X_extra - x_min) / (x_max - x_min)

y_train = torch.tensor(dataset.y_train) - 1
y_test = torch.tensor(dataset.y_test) - 1
y_extra = torch.tensor(dataset.y_400) - 1

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
extra_dataset = TensorDataset(X_extra, y_extra)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
extra_loader = DataLoader(extra_dataset, batch_size=32)

# model = CNN2d().to(device)
model = DeepONet2d().to(device)
# model = FNO2d().to(device)
# model = WNO2d().to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

best_model_error = 99
num_epochs = 100
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

        
        train_loss += loss.item() * inputs.size(0)
        train_abs = torch.abs(outputs - targets)

        train_abs_sum += train_abs.sum().item() / 2
        total_train_samples += inputs.size(0)  

        max_train = train_abs.max().item()
        train_max_diff = max_train if max_train > train_max_diff else train_max_diff
    train_loss /= len(train_loader.dataset)
    scheduler.step()

    # Evaluation
    model.eval()
    test_loss = 0
    test_max_diff = 0
    test_abs_sum = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            test_abs = torch.abs(outputs - targets)
            
            test_abs_sum += test_abs.sum().item() / 2
            test_max = test_abs.max().item()
            test_max_diff = test_max if test_max > test_max_diff else test_max_diff
    test_loss /= len(test_loader.dataset)

    extra_loss = 0
    extra_max_diff = 0
    extra_abs_sum = 0

    with torch.no_grad():
        for inputs, targets in extra_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            extra_loss += loss.item() * inputs.size(0)
            extra_abs = torch.abs(outputs - targets)
            
            extra_abs_sum += extra_abs.sum().item() / 2
            extra_max = extra_abs.max().item()
            extra_max_diff = extra_max if extra_max > extra_max_diff else extra_max_diff
    extra_loss /= len(extra_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} \
          | Loss: {train_loss:.2f} - {test_loss:.2f} - {extra_loss:.2f}\
          | Mean Abs: {train_abs_sum/total_train_samples:.2f} - {test_abs_sum/len(test_loader.dataset):.2f} - {extra_abs_sum/len(extra_loader.dataset):.2f}\
          | Max Abs {train_max_diff:.2f} - {test_max_diff:.2f} - {extra_max_diff:.2f}",
          flush=True)
    
    model_path = Path("./trained")
    model_path.mkdir(exist_ok=True, parents=True)
    test_error = test_abs_sum/len(test_loader.dataset)
    if best_model_error > test_error:
        torch.save(model, model_path / f"{model.name}.pth")
        best_model_error = test_error

print(best_model_error)

'''
CNN:
Epoch 100/100           | Loss: 0.01 - 0.05 - 0.71          | Mean Abs: 0.06 - 0.17 - 0.70          | Max Abs 0.78 - 0.81 - 1.89
0.1469404897093773

DON:
Epoch 100/100           | Loss: 0.01 - 0.03 - 0.51          | Mean Abs: 0.09 - 0.12 - 0.60          | Max Abs 0.72 - 0.69 - 1.73
0.11284642389416695

FNO: 
Epoch 50/50           | Loss: 0.10 - 0.06 - 0.06          | Mean Abs: 0.15 - 0.19 - 0.19          | Max Abs 1.00 - 0.91 - 0.93

'''
