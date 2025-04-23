import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from pathlib import Path
import numpy as np


class DatasetConductivity(Dataset):
    def __init__(self, root="/data/Bei/atrium_conductivity_600", transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.root = Path(root)

        self.folder_path_list = []
        for case_dir in self.root.iterdir():
            for folder_path in case_dir.iterdir():
                if folder_path.is_dir():
                    self.folder_path_list.append(folder_path)

    def len(self):
        return len(self.folder_path_list)

    def get(self, idx):
        folder_path = self.folder_path_list[idx]
        case_path = folder_path.parent

        AT = torch.load(folder_path / "ATs.pt", weights_only=False)
        RT = torch.load(folder_path / "RTs.pt", weights_only=False)

        UAC = torch.from_numpy(np.load(case_path / "UAC.npy"))
        edge_index = torch.load(case_path / "edge_index.pt", weights_only=False)
        y = torch.tensor(y = [float(c) for c in conductivities.name.split("_")])
        
        
        return Data(x=[torch.stack([AT, RT], dim=1), 
                       UAC], 
                    edge_index=edge_index, 
                    y=y.unsqueeze(dim=0))




if __name__ == "__main__":
    dataset = DatasetConductivity()
    dataset.get(2)

