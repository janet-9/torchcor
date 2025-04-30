import torch
import torch.nn as nn
import torch.nn.functional as F

def create_grid_coords(H, W, device='cpu'):
    x = torch.linspace(0, 1, W, device=device)
    y = torch.linspace(0, 1, H, device=device)
    grid_x, grid_y = torch.meshgrid(y, x, indexing='ij') 
    coords = torch.stack([grid_x, grid_y], dim=0)        # (2, H, W)

    return coords

class CNN2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, layers=3):
        super(CNN2d, self).__init__()
        self.name = "cnn_no"

        self.conv_layers = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2))
        self.conv_layers.append(nn.BatchNorm2d(out_channels))
        self.conv_layers.append(nn.ReLU())

        # Subsequent layers
        for _ in range(1, layers):
            self.conv_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=2, padding=2))
            self.conv_layers.append(nn.BatchNorm2d(out_channels))
            self.conv_layers.append(nn.ReLU())

        self.head = nn.Sequential(
            # nn.AdaptiveAvgPool2d((1, 1)),  # (N, 64, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # (N, 64)

            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_channels, 2),
            nn.Tanh()  
        )

    def forward(self, x):
        # coords = create_grid_coords(H=x.size(2), W=x.size(3), device=x.device)
        # coords = coords.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        # x = torch.cat([x, coords], dim=1)
        
        for conv in self.conv_layers:
            x = conv(x)
        
        x = self.head(x)

        return x


if __name__ == "__main__":
    model = CNN2d()
    input_tensor = torch.randn(8, 1, 500, 500)  # N=8
    output = model(input_tensor)
    print(output.shape) 