import torch
import torch.nn as nn
import torch.nn.functional as F

def create_grid_coords(H, W, device='cpu'):
    x = torch.linspace(0, 1, W, device=device)
    y = torch.linspace(0, 1, H, device=device)
    grid_x, grid_y = torch.meshgrid(y, x, indexing='ij') 
    coords = torch.stack([grid_x, grid_y], dim=0)        # (2, H, W)

    return coords

class ConductivityCNN(nn.Module):
    def __init__(self):
        super(ConductivityCNN, self).__init__()
        self.name = "cnn_no"
        
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
            # nn.AdaptiveAvgPool2d((1, 1)),  # (N, 128, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # (N, 128)

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
            nn.Tanh()  
        )

    def forward(self, x):
        # coords = create_grid_coords(H=x.size(2), W=x.size(3), device=x.device)
        # coords = coords.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        # x = torch.cat([x, coords], dim=1)
        
        x = self.conv(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    model = ConductivityCNN()
    input_tensor = torch.randn(8, 1, 500, 500)  # N=8
    output = model(input_tensor)
    print(output.shape) 