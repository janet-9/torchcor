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

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to keep along height
        self.modes2 = modes2  # Number of Fourier modes to keep along width

        self.scale = 1 / (in_channels * out_channels)
        # Low positive frequencies 
        self.weights1 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)  # (in_channels, out_channels, modes1, modes2)
        )
        # Low negative frequencies 
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)  # (in_channels, out_channels, modes1, modes2)
        )

    def compl_mul2d(self, input, weights):
        """
        Args:
            input: (batch, in_channel, height, width)
            weights: (in_channel, out_channel, modes1, modes2)
        Returns:
            (batch, out_channel, height, width)
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, _, H, W = x.shape
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x, norm='ortho')  # (batch, in_channels, H, W//2 + 1)

        # Initialize output in Fourier space
        out_ft = torch.zeros(B, self.out_channels, H, W//2 + 1,    # (batch, out_channels, H, W//2 + 1)
                             device=x.device, dtype=torch.cfloat)

        # Apply weights on the selected modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2],   # (batch, in_channels, modes1, modes2) @ (in_channels, out_channels, modes1, modes2) 
                                                                    self.weights1)                            # = (batch, out_channels, modes1, modes2)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], 
                                                                     self.weights2)
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')  # (batch, out_channels, H, W)
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1=16, modes2=16, width=64, in_channels=1, out_channels=2, depth=4):
        super(FNO2d, self).__init__()
        self.name = "fno"

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.depth = depth

        # Input lifting: project input to high-dimensional space
        self.fc0 = nn.Linear(in_channels, self.width)

        # Fourier layers + pointwise convolutions
        self.spectral_convs = nn.ModuleList()
        self.pointwise_convs = nn.ModuleList()

        for _ in range(depth):
            self.spectral_convs.append(SpectralConv2d(self.width, self.width, self.modes1, self.modes2))
            self.pointwise_convs.append(nn.Conv2d(self.width, self.width, 1)) 

        self.activation = nn.GELU()

        # Output projection
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), # (N, 128, 1, 1)
            nn.Flatten(),  # (N, 128)

            nn.Linear(self.width, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
            nn.Tanh()  
        )

    def forward(self, x):
        # coords = create_grid_coords(H=x.size(2), W=x.size(3), device=x.device)
        # coords = coords.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        # x = torch.cat([x, coords], dim=1)

        # Reshape input: (N, C_in, H, W) -> (N, H, W, C_in)
        x = x.permute(0, 2, 3, 1)

        # Lift to high-dimensional space
        x = self.fc0(x)  # (N, H, W, width)

        # Reshape back for convs: (N, H, W, width) -> (N, width, H, W)
        x = x.permute(0, 3, 1, 2)

        for spectral_conv, pointwise_conv in zip(self.spectral_convs, self.pointwise_convs):
            x1 = spectral_conv(x)
            x2 = pointwise_conv(x)
            x = self.activation(x1 + x2)

        x = self.head(x)

        return x



# class FNOWithGlobalHead(nn.Module):
#     def __init__(self, n_modes=(16, 16), in_channels=3, out_channels=2, hidden_channels=64):
#         super().__init__()
#         self.name = "fno"

#         self.fno = nn.Sequential(
#             FNO(n_modes=n_modes,
#                 n_layers=4,
#                 hidden_channels=hidden_channels,
#                 in_channels=in_channels,   
#                 out_channels=hidden_channels,
#                 positional_embedding="grid",
#                 channel_mlp_dropout=0.1),
#             nn.GroupNorm(4, hidden_channels),
#             nn.ReLU(),

#             # FNO(n_modes=n_modes,
#             #     n_layers=4,
#             #     hidden_channels=hidden_channels,
#             #     in_channels=hidden_channels,  
#             #     out_channels=hidden_channels),
#             # nn.GroupNorm(4, hidden_channels),
#             # nn.ReLU(),                      # (N, 128, 100, 100)
#         )

#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),  # reduce to (N, 128, 1, 1)
#             nn.Flatten(),  # (N, 128)
            
#             nn.Linear(hidden_channels, out_channels),
#             nn.Dropout(0.3),
#             nn.Tanh()  
#         )

#     def forward(self, x):
#         coords = create_grid_coords(H=x.size(2), W=x.size(3), device=x.device)
#         coords = coords.unsqueeze(0).expand(x.size(0), -1, -1, -1)
#         x = torch.cat([x, coords], dim=1)

#         x = self.fno(x)            # Output: (N, 64, 500, 500) 
#         out = self.head(x)         # Output: (N, 2)
#         return out

# if __name__ == "__main__":
#     model = FNOWithGlobalHead()
#     input_tensor = torch.randn(32, 1, 50, 50)  # (batch_size, in_channels, height, width)
#     output = model(input_tensor)
#     # print(output.shape) 
#     print(create_grid_coords(3, 3, "cpu").permute(1, 2, 0))
