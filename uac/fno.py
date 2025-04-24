import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO

class FNOWithGlobalHead(nn.Module):
    def __init__(self, n_modes=(16, 16), in_channels=1, out_channels=2, hidden_channels=64):
        super().__init__()
        self.fno1 = FNO(n_modes=n_modes,
                        n_layers=4,
                        hidden_channels=hidden_channels,
                        in_channels=in_channels,   
                        out_channels=hidden_channels) 
        self.norm1 = nn.GroupNorm(8, hidden_channels)
        
        self.fno2 = FNO(n_modes=n_modes,
                        n_layers=2,
                        hidden_channels=hidden_channels,
                        in_channels=hidden_channels,  
                        out_channels=hidden_channels) 
        self.norm2 = nn.GroupNorm(8, hidden_channels)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # reduce to (N, 64, 1, 1)
            nn.Flatten(),  # (N, 128)
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, out_channels),
            nn.Sigmoid()  
        )

    def forward(self, x):
        # x = self.add_coords(x)
        x = self.fno1(x)            # Output: (N, 64, 500, 500) 
        # x = self.norm1(x) 
        x = F.relu(x)
    
        x = self.fno2(x)
        # x = self.norm2(x)
        x = F.relu(x)

        out = self.head(x)         # Output: (N, 2)
        return out

if __name__ == "__main__":
    model = FNOWithGlobalHead()
    input_tensor = torch.randn(128, 1, 500, 500)  # (batch_size, in_channels, height, width)
    output = model(input_tensor)
    print(output.shape) 
