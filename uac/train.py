import torch
import torch.nn as nn
import torch.optim as optim
from dataset import Dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model import ConductivityCNN

dataset = Dataset(n_uac_points=500)
X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConductivityCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_max_diff = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

        max_train = torch.abs(outputs - targets).max().item()
        train_max_diff = max_train if max_train > train_max_diff else train_max_diff

    train_loss /= len(train_loader.dataset)

    # Evaluation
    model.eval()
    test_loss = 0
    test_max_diff = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

    test_loss /= len(test_loader.dataset)

    max_test = torch.abs(outputs - targets).max().item()
    test_max_diff = max_test if max_test > test_max_diff else test_max_diff

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.2f} | Test Loss: {test_loss:.2f} | Train Max: {train_max_diff:.2f} | Test Max: {test_max_diff:.2f}")
