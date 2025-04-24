import torch
import torch.nn as nn
import torch.optim as optim
from dataset import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model import ConductivityCNN
from fno import FNOWithGlobalHead

dataset = Dataset(n_uac_points=500)
X_train = torch.tensor(dataset.X_train[:, :1, :, :])
X_test = torch.tensor(dataset.X_test[:, :1, :, :])
y_train = torch.tensor(dataset.y_train)
y_test = torch.tensor(dataset.y_test)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# model = ConductivityCNN().to(device)
model = FNOWithGlobalHead().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_mean_diff = []
    for inputs, targets in train_loader:

        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        train_mean = torch.abs(outputs - targets).mean().item()
        train_mean_diff.append(train_mean)
        # train_max_diff = max_train if max_train > train_max_diff else train_max_diff

    train_loss /= len(train_loader.dataset)
    scheduler.step()

    # Evaluation
    model.eval()
    test_loss = 0
    test_mean_diff = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            test_mean = torch.abs(outputs - targets).mean().item()
            test_mean_diff.append(test_mean)
            # test_max_diff = max_test if max_test > test_max_diff else test_max_diff

    test_loss /= len(test_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.2f} | Test Loss: {test_loss:.2f} | Train Max: {sum(train_mean_diff)/len(train_mean_diff):.2f} | Test Max: {sum(test_mean_diff)/len(test_mean_diff):.2f}")
