import torch
import torch.nn as nn
import torch.nn.functional as F

class ConductivityCNN(nn.Module):
    def __init__(self):
        super(ConductivityCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # (N, 16, 250, 250)
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # (N, 32, 125, 125)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # (N, 64, 63, 63)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # (N, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (N, 128, 1, 1)
            nn.Flatten(),  # (N, 128)

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
            nn.Tanh()  
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = ConductivityCNN()
    input_tensor = torch.randn(8, 1, 500, 500)  # N=8
    output = model(input_tensor)
    print(output.shape) 