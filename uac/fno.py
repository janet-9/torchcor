import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO

class ConductivityFNO(nn.Module):
    def __init__(self, modes=24, width=64):
        super(ConductivityFNO, self).__init__()
        self.fno = FNO(
            in_channels=2,
            out_channels=2,
            dimension=2,
            n_modes=(modes, modes),
            hidden_channels=width,
        )

        self.output_activation = nn.Tanh()  # match your output range

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fno(x)
        x = x.mean(dim=(1, 2))  # global average pooling over spatial dims
        x = self.output_activation(x)
        return x

if __name__ == "__main__":
    model = ConductivityFNO()
    input_tensor = torch.randn(8, 2, 500, 500)  # N=8
    output = model(input_tensor)
    print(output.shape) 