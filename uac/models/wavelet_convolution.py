import numpy as np
import torch
import torch.nn as nn
from pytorch_wavelets import DWT, IDWT 


""" Def: 2d Wavelet convolutional layer (discrete) """
class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet="db6", mode='periodization'):
        super(WaveConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.size = size

        self.wavelet = wavelet
        self.mode = mode
        dummy_data = torch.randn(1, 1, *self.size)        
        dwt_ = DWT(J=self.level, mode=self.mode, wave=self.wavelet)
        mode_data, mode_coef = dwt_(dummy_data)
        self.modes1 = mode_data.shape[-2]
        self.modes2 = mode_data.shape[-1]
        
        # Parameter initilization
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Convolution
    def mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        if x.shape[-1] > self.size[-1]:
            factor = int(np.log2(x.shape[-1] // self.size[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level+factor, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
            
        elif x.shape[-1] < self.size[-1]:
            factor = int(np.log2(self.size[-1] // x.shape[-1]))
            
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level-factor, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)
        
        else:
            # Compute single tree Discrete Wavelet coefficients using some wavelet
            dwt = DWT(J=self.level, mode=self.mode, wave=self.wavelet).to(x.device)
            x_ft, x_coeff = dwt(x)

        # Instantiate higher level coefficients as zeros
        out_ft = torch.zeros_like(x_ft, device= x.device)
        out_coeff = [torch.zeros_like(coeffs, device= x.device) for coeffs in x_coeff]
        
        # Multiply the final approximate Wavelet modes
        out_ft = self.mul2d(x_ft, self.weights1)
        # Multiply the final detailed wavelet coefficients
        out_coeff[-1][:,:,0,:,:] = self.mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        out_coeff[-1][:,:,1,:,:] = self.mul2d(x_coeff[-1][:,:,1,:,:].clone(), self.weights3)
        out_coeff[-1][:,:,2,:,:] = self.mul2d(x_coeff[-1][:,:,2,:,:].clone(), self.weights4)
        
        # Return to physical space        
        idwt = IDWT(mode=self.mode, wave=self.wavelet).to(x.device)
        x = idwt((out_ft, out_coeff))

        return x

if __name__ == "__main__":
    wavelet_layer = WaveConv2d(in_channels=10, 
                               out_channels=10, 
                               level=2, 
                               size=[100, 100], 
                               wavelet="db6", 
                               mode="periodization")  # symmetric
    

    input_tensor = torch.randn(32, 10, 50, 50) 
    print(wavelet_layer(input_tensor).shape)

    input_tensor = torch.randn(32, 10, 200, 200) 
    print(wavelet_layer(input_tensor).shape)

    input_tensor = torch.randn(32, 10, 400, 400) 
    print(wavelet_layer(input_tensor).shape)

    # wavelet_layer.reset_size(size=[50, 50])
    # input_tensor = torch.randn(32, 1, 50, 50) 
    # print(wavelet_layer(input_tensor).shape)