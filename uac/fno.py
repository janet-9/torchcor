import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO

class FNOWithGlobalHead(nn.Module):
    def __init__(self, n_modes=(32, 32), in_channels=1, out_channels=2, hidden_channels=64):
        super().__init__()
        self.fno = nn.Sequential(
            FNO(n_modes=n_modes,
                n_layers=4,
                hidden_channels=hidden_channels,
                in_channels=in_channels,   
                out_channels=hidden_channels,
                channel_mlp_dropout=0.1),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(),

            # FNO(n_modes=n_modes,
            #     n_layers=2,
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
        # x = self.add_coords(x)
        x = self.fno(x)            # Output: (N, 64, 500, 500) 
        out = self.head(x)         # Output: (N, 2)
        return out

if __name__ == "__main__":
    model = FNOWithGlobalHead()
    input_tensor = torch.randn(128, 1, 500, 500)  # (batch_size, in_channels, height, width)
    output = model(input_tensor)
    print(output.shape) 
