import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT, IDWT

class FlexibleWaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, wavelet="db1", mode="symmetric"):
        super(FlexibleWaveConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.wavelet = wavelet
        self.mode = mode

        # We define smaller weight parameters (say 16x16 modes)
        self.base_modes = (16, 16)   # you can adjust this
        self.scale = (1 / (in_channels * out_channels))

        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, *self.base_modes))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, *self.base_modes))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, *self.base_modes))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, *self.base_modes))

    def mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def interpolate_weights(self, weights, target_shape):
        # Bilinear interpolation to match target mode size
        weights = F.interpolate(weights, size=target_shape, mode="bilinear", align_corners=False)
        return weights

    def forward(self, x):
        batchsize, channels, height, width = x.shape

        # Compute DWT
        dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet).to(x.device)
        x_ft, x_coeff = dwt(x)

        # Dynamically resize weights
        target_shape = (x_ft.shape[-2], x_ft.shape[-1])
        weights1 = self.interpolate_weights(self.weights1, target_shape)
        weights2 = self.interpolate_weights(self.weights2, target_shape)
        weights3 = self.interpolate_weights(self.weights3, target_shape)
        weights4 = self.interpolate_weights(self.weights4, target_shape)

        # Allocate output coefficients
        out_ft = torch.zeros_like(x_ft, device=x.device)
        out_coeff = [torch.zeros_like(coeffs, device=x.device) for coeffs in x_coeff]

        # Apply wavelet convolution
        out_ft = self.mul2d(x_ft, weights1)

        # Detailed coefficients
        out_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), weights2)
        out_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), weights3)
        out_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), weights4)

        # Inverse DWT
        idwt = IDWT(mode=self.mode, wave=self.wavelet).to(x.device)
        x = idwt((out_ft, out_coeff))

        return x

if __name__ == "__main__":
    wavelet_layer = FlexibleWaveConv2d(in_channels=1, out_channels=1, level=2)

    input_tensor = torch.randn(32, 1, 50, 50)
    output = wavelet_layer(input_tensor)
    print(output.shape)  # (32, 1, 50, 50)

    input_tensor = torch.randn(32, 1, 100, 100)
    output = wavelet_layer(input_tensor)
    print(output.shape)  # (32, 1, 100, 100)

    input_tensor = torch.randn(32, 1, 128, 128)
    output = wavelet_layer(input_tensor)
    print(output.shape)  # (32, 1, 128, 128)
