import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO

def create_grid_coords(H, W, device='cpu'):
    x = torch.linspace(0, 1, W, device=device)
    y = torch.linspace(0, 1, H, device=device)
    grid_x, grid_y = torch.meshgrid(y, x, indexing='ij') 
    coords = torch.stack([grid_x, grid_y], dim=0)        # (2, H, W)

    return coords

class FNOWithGlobalHead(nn.Module):
    def __init__(self, n_modes=(16, 16), in_channels=3, out_channels=2, hidden_channels=64):
        super().__init__()
        self.name = "fno"

        self.fno = nn.Sequential(
            FNO(n_modes=n_modes,
                n_layers=4,
                hidden_channels=hidden_channels,
                in_channels=in_channels,   
                out_channels=hidden_channels,
                positional_embedding="grid",
                channel_mlp_dropout=0.1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(),

            # FNO(n_modes=n_modes,
            #     n_layers=4,
            #     hidden_channels=hidden_channels,
            #     in_channels=hidden_channels,  
            #     out_channels=hidden_channels),
            # nn.GroupNorm(4, hidden_channels),
            # nn.ReLU(),                      # (N, 128, 100, 100)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # reduce to (N, 128, 1, 1)
            nn.Flatten(),  # (N, 128)
            
            nn.Linear(hidden_channels, out_channels),
            nn.Dropout(0.3),
            nn.Tanh()  
        )

    def forward(self, x):
        coords = create_grid_coords(H=x.size(2), W=x.size(3), device=x.device)
        coords = coords.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        x = torch.cat([x, coords], dim=1)

        x = self.fno(x)            # Output: (N, 64, 500, 500) 
        out = self.head(x)         # Output: (N, 2)
        return out

if __name__ == "__main__":
    model = FNOWithGlobalHead()
    input_tensor = torch.randn(32, 1, 50, 50)  # (batch_size, in_channels, height, width)
    output = model(input_tensor)
    # print(output.shape) 
    print(create_grid_coords(3, 3, "cpu").permute(1, 2, 0))
