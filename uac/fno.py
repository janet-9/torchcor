import torch
import torch.nn as nn
from neuralop.models import FNO

class FNOWithGlobalHead(nn.Module):
    def __init__(self, n_modes=(16, 16), in_channels=1, out_channels=2, hidden_channels=128):
        super().__init__()
        self.fno1 = FNO(n_modes=n_modes,
                        n_layers=4,
                        hidden_channels=hidden_channels,
                        in_channels=in_channels,   # 2 input channels
                        out_channels=hidden_channels) # use more channels for internal representation
        self.norm1 = nn.InstanceNorm2d(hidden_channels)
        
        self.fno2 = FNO(n_modes=n_modes,
                        hidden_channels=hidden_channels,
                        in_channels=hidden_channels,   # 2 input channels
                        out_channels=hidden_channels) # use more channels for internal representation
        self.norm2 = nn.InstanceNorm2d(hidden_channels)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # reduce to (N, 64, 1, 1)
            nn.Flatten(),  # (N, 128)
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels),
            nn.Sigmoid()  # Output in [-1, 1]
        )

    def add_coords(self, x):
        # Add normalized coordinate grid: shape (N, C+2, H, W)
        N, C, H, W = x.shape
        y_coords = torch.linspace(0, 1, H, device=x.device).view(1, 1, H, 1).expand(N, 1, H, W)
        x_coords = torch.linspace(0, 1, W, device=x.device).view(1, 1, 1, W).expand(N, 1, H, W)
        return torch.cat([x, x_coords, y_coords], dim=1)

    def forward(self, x):
        # x = self.add_coords(x)
        x = self.fno1(x)            # Output: (N, 64, 500, 500) 
        # x = self.norm1(x) 
    

        # x = self.fno2(x)
        # x = self.norm2(x)

        out = self.head(x)         # Output: (N, 2)
        return out

if __name__ == "__main__":
    model = FNOWithGlobalHead()
    input_tensor = torch.randn(128, 1, 500, 500)  # (batch_size, in_channels, height, width)
    output = model(input_tensor)
    print(output.shape) 
