import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from pathlib import Path
import numpy as np


class DatasetBranch(Dataset):
    def __init__(self, at_rt_root="/data/Bei/atrium_conductivity_2", uac_root="/data/Bei/meshes_refined", transform=None, pre_transform=None):
        super().__init__(at_rt_root, transform, pre_transform)
        self.at_rt_root = Path(at_rt_root)
        self.uac_root = Path(uac_root)

        self.at_rt_list = []
        for case_dir in self.root.iterdir():
            for file_path in case_dir.iterdir():
                at_path = file_path / "ATs.pt"
                rt_path = file_path / "RTs.pt"
                self.file_list.append((at_path, rt_path))
        
        self.uac_list = []
        for case_dir in self.root.iterdir():
            uac = case_dir / "UAC.npy"
            self.uac_list.append(uac)

        def len(self):
            return len(self.file_list)

        def get(self, idx):
            at_path, rt_path = self.file_list[idx]
            AT = torch.load(at_path)
            RT = torch.load(rt_path)

            UAC = torch.load()





if __name__ == "__main__":
    dataset = ConductivityDataset()

