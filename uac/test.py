import torch
import torch.nn as nn

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
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        # Low negative frequencies 
        self.weights2 = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
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
        batchsize, size_x, size_y = x.shape[0], x.shape[-2], x.shape[-1]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x, norm='ortho')  # (batch, in_channels, size_x, size_y//2 + 1)

        # Initialize output in Fourier space
        out_ft = torch.zeros(batchsize, self.out_channels, size_x, size_y//2 + 1,
                             device=x.device, dtype=torch.cfloat)

        # Apply weights on the selected modes
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], 
                                                                    self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], 
                                                                     self.weights2)
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(size_x, size_y), norm='ortho')
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels=1, out_channels=2, depth=4):
        super(FNO2d, self).__init__()

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
            # nn.AdaptiveAvgPool2d((1, 1)),  # (N, 128, 1, 1)
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),  # (N, 128)

            nn.Linear(self.width, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
            nn.Tanh()  
        )

    def forward(self, x):
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

if __name__ == "__main__":
    model = FNO2d(modes1=16, modes2=16, width=32)

    x = torch.randn(8, 1, 80, 80)  # Batch of 8
    out = model(x)

    print(out.shape)  # -> (8, 2)

