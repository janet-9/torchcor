import torch
import torch.nn as nn
import torch.nn.functional as F



class BranchNet(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (N,16,50,50)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # (N,32,50,50)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (N,64,50,50)
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),               # (N,64,1,1)
            nn.Flatten(),                               # (N,64)
            nn.Dropout(0.2),
            nn.Linear(64, latent_dim),                  # (N,latent_dim)
            nn.ReLU()
        )

    def forward(self, x):  # (N, 1, 50, 50)
        x = self.conv(x)
        x = self.head(x)
        return x  # (N, latent_dim)

class TrunkNet(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),

            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
    
    def create_grid_coords(self, H, W, device='cpu'):
        x = torch.linspace(0, 1, W, device=device)
        y = torch.linspace(0, 1, H, device=device)
        grid_x, grid_y = torch.meshgrid(y, x, indexing='ij') 

        coords = torch.stack([grid_x, grid_y], dim=-1)        # (H, W, 2)
        coords = coords.view(-1, 2)                           # (H*W, 2)

        return coords

    def forward(self, x):  # coords: (num_points, 2)
        coords = self.create_grid_coords(H=x.size(2), W=x.size(3), device=x.device)
        
        coords = self.mlp(coords)         # (num_points, latent_dim)
        coords = coords.mean(dim=0)       # (latent_dim,)
        out = coords.unsqueeze(0).expand(x.size(0), -1)  # (N, latent_dim)
        return out

class DeepONet(nn.Module):
    def __init__(self, latent_dim=64, output_dim=2):
        super().__init__()
        self.name = "don"
        self.branch_net = BranchNet(latent_dim)
        self.trunk_net = TrunkNet(latent_dim)
    
        self.head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),

            nn.Linear(64, output_dim),
            nn.Tanh() 
        )

    def forward(self, x):
        branch_out = self.branch_net(x)               # (N, latent_dim)
        trunk_out = self.trunk_net(x)                 # (N, latent_dim)

        x = branch_out * trunk_out
        x = self.head(x)

        return x

# Example usage
if __name__ == "__main__":
    model = DeepONet(latent_dim=64, output_dim=2)
    input_tensor = torch.randn(8, 1, 50, 50)  # (batch_size, channels, height, width)
    output = model(input_tensor)
    print(output.shape)  # should be (8, 2)
