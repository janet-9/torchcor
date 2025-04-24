import torch
from torch.utils.data import Dataset
from pathlib import Path
from scipy.interpolate import griddata
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import Manager

class Dataset(Dataset):
    def __init__(self, n_uac_points=100, root="/data/Bei/"):
        self.n_uac_points = n_uac_points
        self.root = Path(root)
        self.mesh_dir = self.root / "meshes_refined"
        self.at_rt_dir = self.root / "atrium_conductivity_2"
        self.dataset_path = self.root / f"dataset_{self.n_uac_points}"
        
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.X_extra = []
        self.y_extra = []

        if not self.dataset_path.exists():
            self.dataset_path.mkdir(exist_ok=True, parents=True)
            with Manager() as manager:
                X_list = manager.list()
                y_list = manager.list()
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
                        for future in as_completed(futures):
                            X, y = future.result()
                            X_list.append(X)
                            y_list.append(y)

                        np.savez(self.dataset_path / f"{case}.npz", 
                                 X=np.stack(X_list).astype(np.float32), 
                                 y=np.stack(y_list).astype(np.float32))
                        X_list[:] = []
                        y_list[:] = []

                        
        self.load_data()

    def load_data(self):
        for i in range(1, 101):
            data = np.load(self.dataset_path / f"Case_{i}.npz")
            X = data['X'].astype(np.float32)
            y = data['y'].astype(np.float32) / 2
            
            if i <= 90:
                self.X_train.append(X)
                self.y_train.append(y)
            else:
                self.X_test.append(X)
                self.y_test.append(y)

        self.X_train = np.concatenate(self.X_train, axis=0)
        self.y_train = np.concatenate(self.y_train, axis=0)
        self.X_test = np.concatenate(self.X_test, axis=0)
        self.y_test = np.concatenate(self.y_test, axis=0)

        x_min = self.X_train.min()
        x_max = self.X_train.max()
        self.X_train = (self.X_train - x_min) / (x_max - x_min)
        self.X_test = (self.X_test - x_min) / (x_max - x_min)

        for i in range(91, 101):
            data = np.load(self.root / f"dataset_300" / f"Case_{i}.npz")
            X = data['X'].astype(np.float32)
            y = data['y'].astype(np.float32) / 2
            self.X_extra.append(X)
            self.y_extra.append(y)

        self.X_extra = np.concatenate(self.X_extra, axis=0)
        self.y_extra = np.concatenate(self.y_extra, axis=0)
        self.X_extra = (self.X_extra - x_min) / (x_max - x_min)

        print("Completed loading data", flush=True)


    def process_case(self, UAC, at_rt_path, n_uac_points):
        AT = torch.load(at_rt_path / "ATs.pt", weights_only=False).numpy().astype(np.float32)
        RT = torch.load(at_rt_path / "RTs.pt", weights_only=False).numpy().astype(np.float32)
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

        return uac_values.astype(np.float32)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y

if __name__ == "__main__":
    d = Dataset()