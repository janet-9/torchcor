import torch
import torch.nn as nn
import torch.optim as optim
from dataset import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from cnn import ConductivityCNN
from fno import FNOWithGlobalHead
import argparse

parser = argparse.ArgumentParser(description="root",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-root", type=str, default="/data/Bei")
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_id = torch.cuda.current_device()
gpu_name = torch.cuda.get_device_name(device_id)
print(gpu_name, flush=True)

dataset = Dataset(n_uac_points=100, root=args.root)
X_train = torch.tensor(dataset.X_train[:, :1, :, :])
X_test = torch.tensor(dataset.X_test[:, :1, :, :])
X_extra = torch.tensor(dataset.X_extra[:, :1, :, :])
y_train = torch.tensor(dataset.y_train)
y_test = torch.tensor(dataset.y_test)
y_extra = torch.tensor(dataset.y_extra)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
extra_dataset = TensorDataset(X_extra, y_extra)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
extra_loader = DataLoader(extra_dataset, batch_size=32)

# model = ConductivityCNN().to(device)
model = FNOWithGlobalHead().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)

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
        train_abs = torch.abs(outputs * 2 - targets * 2)
        train_abs_sum += train_abs.sum().item()  
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
    total_test_samples = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            test_abs = torch.abs(outputs * 2 - targets * 2)
            
            test_abs_sum += test_abs.sum().item()  
            total_test_samples += inputs.size(0) 
            test_max = test_abs.max().item()
            test_max_diff = test_max if test_max > test_max_diff else test_max_diff
    test_loss /= len(test_loader.dataset)

    extra_loss = 0
    extra_max_diff = 0
    extra_abs_sum = 0
    total_extra_samples = 0
    with torch.no_grad():
        for inputs, targets in extra_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            extra_loss += loss.item() * inputs.size(0)
            extra_abs = torch.abs(outputs * 2 - targets * 2)
            
            extra_abs_sum += extra_abs.sum().item()  
            total_extra_samples += inputs.size(0) 
            extra_max = extra_abs.max().item()
            extra_max_diff = extra_max if extra_max > extra_max_diff else extra_max_diff
    extra_loss /= len(extra_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} \
          | Loss: {train_loss:.2f} - {test_loss:.2f} - {extra_loss:.2f}\
          | Mean Abs: {train_abs_sum/total_train_samples:.2f} - {test_abs_sum/total_test_samples:.2f} - {extra_abs_sum/total_extra_samples:.2f}\
          | Max Abs {train_max_diff:.2f} - {test_max_diff:.2f} - {extra_max_diff:.2f}",
          flush=True)
