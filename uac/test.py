import torch
import torch.nn as nn
import torch.fft

class FourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes=16):
        super(FourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # number of low-frequency modes to keep

        # This is the linear transformation applied in Fourier space
        self.scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes, modes))
        self.weights_imag = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes, modes))

    def forward(self, x):
        '''
        x: (batch, height, width, in_channels)
        '''
        batchsize, h, w, c = x.shape
        x = x.permute(0, 3, 1, 2)  # (batch, in_channels, height, width)

        # Fourier Transform
        x_ft = torch.fft.rfft2(x, norm='ortho')

        # Initialize output in Fourier domain
        out_ft = torch.zeros(batchsize, self.out_channels, h, w//2+1, dtype=torch.cfloat, device=x.device)

        # Apply learned weights on low-frequency modes
        for i in range(self.in_channels):
            for j in range(self.out_channels):
                out_ft[:, j, :self.modes, :self.modes] += (
                    x_ft[:, i, :self.modes, :self.modes] * 
                    (self.weights_real[i, j] + 1j * self.weights_imag[i, j])
                )
        
        # Inverse Fourier Transform
        x = torch.fft.irfft2(out_ft, s=(h, w), norm='ortho')
        raise Exception(x.shape)
        x = x.permute(0, 2, 3, 1)  # (batch, height, width, out_channels)
        return x

class SimpleFNO(nn.Module):
    def __init__(self, modes=16):
        super(SimpleFNO, self).__init__()
        self.fourier = FourierLayer(in_channels=1, out_channels=16, modes=modes)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * 50 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        '''
        x: (batch, 50, 50, 1)
        '''
        x = self.fourier(x)
        x = self.fc(x)
        return x
    

model = SimpleFNO(modes=16)
x = torch.randn(8, 50, 50, 1)  # batch of 8
out = model(x)
print(out.shape)  # should be (8, 2)

