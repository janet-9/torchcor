import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.interpolate import griddata
import numpy as np
from concurrent.futures import ProcessPoolExecutor

class Dataset(Dataset):
    def __init__(self, n_uac_points=500, mesh_dir="/data/Bei/meshes_refined/", at_rt_dir="/data/Bei/atrium_conductivity_2"):
        self.n_uac_points = n_uac_points
        self.mesh_dir = Path(mesh_dir)
        self.at_rt_dir = Path(at_rt_dir)
        self.dataset_path = Path(f"./dataset_{self.n_uac_points}.npz")

        if not self.dataset_path.exists():
            self.X = None
            self.y = None

            X_list = []
            y_list = []
            with ProcessPoolExecutor() as executor:
                for i in range(1, 101):
                    case = f"Case_{i}"
                    print(case)
                    uac_path = self.mesh_dir / case / "UAC.npy"
                    UAC = torch.from_numpy(np.load(uac_path))
                    futures = []
                    for at_rt_path in (self.at_rt_dir / case).iterdir():
                        future = executor.submit(self.process_case, UAC, at_rt_path, n_uac_points)
                        futures.append(future)
                    for future in futures:
                        X, y = future.result()
                        X_list.append(X)
                        y_list.append(y)
                    print(len(X_list), len(y_list))

            self.X = np.stack(X_list)
            self.y = np.stack(y_list)

            self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
            self.y = self.y - 1

            np.savez(self.dataset_path, X=self.X, y=self.y)

        else:
            data = np.load(self.dataset_path)
            self.X = data['X']
            self.y = data['y']

    def process_case(self, UAC, at_rt_path, n_uac_points):
        AT = torch.load(at_rt_path / "ATs.pt", weights_only=False).numpy()
        RT = torch.load(at_rt_path / "RTs.pt", weights_only=False).numpy()
        X = self.uac_interpolate(UAC, AT, RT, n_uac_points)
        y = [float(c) for c in at_rt_path.name.split("_")]

        return X, y

    def uac_interpolate(self, uac, at, rt, n_uac_points):
        grid = np.linspace(0, 1, n_uac_points)
        grid_points = np.meshgrid(grid, grid)

        grid_ats_linear = griddata(uac, at, (grid_points[0], grid_points[1]), method='linear', fill_value=np.nan)
        grid_ats_nearest = griddata(uac, at, (grid_points[0], grid_points[1]), method='nearest')
        grid_ats = np.where(np.isnan(grid_ats_linear),
                            grid_ats_nearest,
                            grid_ats_linear)
        
        grid_rts_linear = griddata(uac, rt, (grid_points[0], grid_points[1]), method='linear', fill_value=np.nan)
        grid_rts_nearest = griddata(uac, rt, (grid_points[0], grid_points[1]), method='nearest')
        grid_rts = np.where(np.isnan(grid_rts_linear),
                            grid_rts_nearest,
                            grid_rts_linear)

        uac_values = np.stack([grid_ats, grid_rts], axis=0)

        return uac_values
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y

if __name__ == "__main__":
    d = Dataset()