import sys
import os
import warnings
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo
import time
from core import *


class Monodomain:
    def __init__(self, ionic_model, T, dt, device=None, dtype=None):
        self.device = torch.device(device) if device is not None else "cuda:0" \
            if torch.cuda.is_available() else "cpu"
        self.dtype = dtype if dtype is not None else torch.float64

        self.T = T  # ms
        self.dt = dt  # ms
        self.nt = int(T // dt)

        self.ionic_model = ionic_model

        self.pcd = None
        self.cg = None

        self.n_nodes = None
        self.nodes = None
        self.elems = None
        self.fibres = None
        self.region_ids = None

        self.stimuli = None
        self.conductivity = None

        self.K = None
        self.M = None
        self.A = None
        self.Chi = 140 
        self.Cm = 0.01  
        self.theta = 0.5

        nvmlInit()
        device_id = torch.cuda.current_device()
        self.gpu_handle = nvmlDeviceGetHandleByIndex(device_id)


    def load_mesh(self, path="/Users/bei/Project/FinitePDE/data/Case_1", unit_conversion=1000):
        reader = MeshReader(path)
        nodes, elems, regions, fibres = reader.read(unit_conversion=unit_conversion)
        
        self.n_nodes = nodes.shape[0]
        self.nodes = torch.from_numpy(nodes).to(dtype=self.dtype, device=self.device)
        self.elems = torch.from_numpy(elems).to(dtype=torch.int, device=self.device)
        self.regions = torch.from_numpy(regions).to(dtype=torch.int, device=self.device)
        self.fibres = torch.from_numpy(fibres).to(dtype=self.dtype, device=self.device)

        self.stimuli = Stimuli(self.n_nodes, self.device, self.dtype)
        self.conductivity = Conductivity(self.regions, dtype=self.dtype)

    def add_stimulus(self, vtx_filepath, start, duration, intensity, period=1, count=1):
        self.stimuli.add(vtx_filepath, start, duration, intensity, period, count)

    def add_condutivity(self, region_ids, il, it, el=None, et=None):
        self.conductivity.add(region_ids, il, it, el, et)

    def assemble(self):
        sigma = self.conductivity.calculate_sigma(self.fibres)

        if self.elems.shape[1] == 3:
            matrices = Matrices3DSurface(vertices=self.nodes, triangles=self.elems, device=self.device, dtype=self.dtype)
        else:
            matrices = Matrices3D(vertices=self.nodes, tetrahedrons=self.elems, device=self.device, dtype=self.dtype)

        K, M = matrices.assemble_matrices(sigma)

        self.K = K.to(device=self.device, dtype=self.dtype)
        self.M = M.to(device=self.device, dtype=self.dtype)
        A = self.M * self.Cm * self.Chi + self.K * self.dt * self.theta
        A = A.coalesce()

        self.pcd = Preconditioner()
        self.pcd.create_Jocobi(A)

        self.M = self.M.to_sparse_csr()
        self.K = self.K.to_sparse_csr()
        self.A = A.to_sparse_csr()

        self.cg = ConjugateGradient(self.pcd, self.A, dtype=torch.float64)
        

    def step(self, u, t, a_tol, r_tol, max_iter):
        du = self.ionic_model.differentiate(u) / 100
        b = u * self.Cm + self.dt * du

        Istim = self.stimuli.apply(t) / self.Chi
        b += self.dt * Istim

        b = self.Chi * self.M @ b
        b -= (1 - self.theta) * self.dt * self.K @ u

        u, n_iter = self.cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
        
        return u, n_iter

    def solve(self, a_tol, r_tol, max_iter, plot_interval=10, verbose=True):
        u = self.ionic_model.initialize(self.n_nodes)
        gpu_utilisation_list = []
        gpu_memory_list = []

        self.cg.initialize(x=u)

        ts_per_frame = int(plot_interval / self.dt)
        if self.elems.shape[1] == 3:
            visualization = VTK3DSurface(self.nodes.cpu().numpy(), self.elems.cpu().numpy())
        else:
            visualization = VTK3D(self.nodes.cpu().numpy(), self.elems.cpu().numpy())

        t = 0
        solving_time = time.time()
        n_total_iter = 0
        for n in range(1, self.nt + 1):
            t += self.dt
            
            u, n_iter = self.step(u, t, a_tol, r_tol, max_iter)
            n_total_iter += n_iter
            # print(f"{round(t, 2)} / {self.T}: {n_iter}")
            # if n % ts_per_frame == 0:
            #     visualization.save_frame(color_values=u.cpu().numpy(),
            #                              frame_path=f"./vtk_{self.n_nodes}/frame_{n}.vtk")

            if n % ts_per_frame == 0:
                gpu_utilisation_list.append(nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu)
                gpu_memory_list.append(nvmlDeviceGetMemoryInfo(self.gpu_handle).used / 1e9)

        print(self.n_nodes, 
              round(time.time() - solving_time, 2),
              n_total_iter,
              f"{round(sum(gpu_utilisation_list)/len(gpu_utilisation_list), 2)}",
              f"{round(sum(gpu_memory_list)/len(gpu_memory_list), 2)}")
        # print(f"Ran {n_total_iter} iterations in {round(time.time() - solving_time, 2)} seconds;")

    def save(self, dir, format="igb"):
        pass
        

if __name__ == "__main__":
    from Projects.torchcor.model.monodomain import Monodomain
    from ionic import TenTusscherPanfilov
    import torch
    from pathlib import Path

    simulation_time = 1000
    dt = 0.01

    device = torch.device(f"cuda:2" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    home_dir = Path.home()

    ionic_model = TenTusscherPanfilov(cell_type="ENDO", dt=dt, device=device, dtype=dtype)
    simulator = Monodomain(ionic_model, T=simulation_time, dt=dt, device=device, dtype=dtype)
    simulator.load_mesh(path=f"{home_dir}/Data/ventricle/")
    simulator.add_condutivity([34, 35], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
    simulator.add_condutivity([44, 45, 46], il=0.9074, it=0.3332, el=0.9074, et=0.3332)

    simulator.add_stimulus(f"{home_dir}/Data/ventricle/LV_sf.vtx", start=0.0, duration=1.0, intensity=100, period=500, count=2)
    simulator.add_stimulus(f"{home_dir}/Data/ventricle/LV_pf.vtx", start=0.0, duration=1.0, intensity=100, period=500, count=2)
    simulator.add_stimulus(f"{home_dir}/Data/ventricle/LV_af.vtx", start=0.0, duration=1.0, intensity=100, period=500, count=2)
    simulator.add_stimulus(f"{home_dir}/Data/ventricle/RV_sf.vtx", start=5.0, duration=1.0, intensity=100, period=500, count=2)
    simulator.add_stimulus(f"{home_dir}/Data/ventricle/RV_mod.vtx", start=5.0, duration=1.0, intensity=100, period=500, count=2)

    simulator.assemble()
    simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=1000, plot_interval=10, verbose=True)

