import torch
import torch.nn as nn
from neuralop.models import FNO

class FNOWithGlobalHead(nn.Module):
    def __init__(self, n_modes=(16, 16), hidden_channels=16):
        super().__init__()
        self.fno = FNO(n_modes=n_modes,
                       hidden_channels=hidden_channels,
                       in_channels=1,   # 2 input channels
                       out_channels=64) # use more channels for internal representation

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # reduce to (N, 64, 1, 1)
            nn.Flatten(),                  # (N, 64)
            nn.Linear(64, 2),              # (N, 2)
            nn.Tanh()                      # Optional: restrict to [-1, 1] range
        )

    def forward(self, x):
        x = self.fno(x)            # Output: (N, 64, 500, 500)  
        out = self.head(x)         # Output: (N, 2)
        return out

if __name__ == "__main__":
    model = FNOWithGlobalHead()
    input_tensor = torch.randn(128, 1, 500, 500)  # (batch_size, in_channels, height, width)
    output = model(input_tensor)
    print(output.shape) 
