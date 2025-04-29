import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wavelet_convolution import WaveConv2d


class WNO2d(nn.Module):
    def __init__(self, width=64, level=3, layers=2, size=[100, 100], wavelet="haar", in_channel=3, grid_range=[0, 1]):
        super(WNO2d, self).__init__()
        self.name = "wno"

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range 
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range(self.layers):
            self.conv.append(WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet))
            self.w.append(nn.Conv2d(self.width, self.width, 1))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), # (B, self.width, 1, 1)
            nn.Flatten(),                             # (B, self.width)

            nn.Linear(self.width, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
            nn.Tanh()  
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)                # (B, C, H, W) -> (B, H, W, C)
        grid = self.get_grid(x.shape, x.device)  # (B, H, W, 2)
        x = torch.cat((x, grid), dim=-1)    

        x = self.fc0(x)                      # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)            # (B, C, H, W)

        for index, (convl, wl) in enumerate(zip(self.conv, self.w)):
            x = convl(x) + wl(x)  
            x = F.mish(x)                    # (B, C, H, W)
        
        x = self.head(x)
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]

        gridx = torch.linspace(0, self.grid_range[0], size_x)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])

        gridy = torch.linspace(0, self.grid_range[1], size_y)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])

        return torch.cat((gridx, gridy), dim=-1).to(device)


if __name__ == "__main__":
    wno = WNO2d(width=20, level=2, layers=2, size=[100, 100], wavelet="sym3", in_channel=3, grid_range=[0, 1])
    input_tensor = torch.randn(32, 1, 100, 100) 
    print(wno(input_tensor).shape)

    input_tensor = torch.randn(32, 1, 50, 50) 
    print(wno(input_tensor).shape)

    input_tensor = torch.randn(32, 1, 200, 200) 
    print(wno(input_tensor).shape)

    input_tensor = torch.randn(32, 1, 400, 400) 
    print(wno(input_tensor).shape)