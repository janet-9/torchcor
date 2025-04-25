from dataset import load_data
import torch
import numpy as np
from cnn import ConductivityCNN
from fno import FNOWithGlobalHead

checkpoint = torch.load("./ConductivityCNN.pth")
weights = checkpoint['model_state_dict']
train_min = checkpoint['train_min']
train_max = checkpoint['train_max']
print(train_min, train_max)

model = ConductivityCNN()
model.load_state_dict(weights)
model.eval()

X, y = load_data("/data/Bei/dataset_300/Case_98.npz")

ATs = (torch.tensor(X[:, :1, :, :], dtype=torch.float32) - train_min) / (train_max - train_min)
pred = model(ATs).detach().numpy() + 1

print(np.abs(pred - y).mean())