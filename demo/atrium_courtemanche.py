import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from core.assemble import Matrices3DSurface
from core.preconditioner import Preconditioner
from core.solver import ConjugateGradient
from core.visualize import VTK3DSurface
from core.reorder import RCM as RCM
import time
from mesh.triangulation import Triangulation
from mesh.materialproperties import MaterialProperties
from mesh.stimulus import Stimulus
from monodomain.tools import load_stimulus_region
import numpy as np


class AtriumSimulatorCourtemanche:
    def __init__(self, ionic_model, T, dt, apply_rcm, device=None, dtype=torch.float64):
        self.device = torch.device(device) if device is not None else "cuda:0" \
            if torch.cuda.is_available() else "cpu"
        self.dtype = dtype

        self.T = T  # ms = 2.4s
        self.dt = dt  # ms
        self.nt = int(T // dt)
        self.rcm = RCM(device=device, dtype=dtype) if apply_rcm else None

        self.ionic_model = ionic_model
        self.ionic_model.construct_tables()

        self.pcd = None

        self.point_region_ids = None
        self.n_nodes = None
        self.vertices = None
        self.triangles = None
        self.fibers = None

        self.material_config = None

        self.K = None
        self.M = None
        self.A = None

        self.stimulus_region = None
        self.stimuli = []

    def load_mesh(self, path="/Users/bei/Project/FinitePDE/data/Case_1"):
        mesh = Triangulation()
        mesh.readMesh(path)

        self.point_region_ids = mesh.point_region_ids()
        self.n_nodes = self.point_region_ids.shape[0]

        self.vertices = torch.from_numpy(mesh.Pts()).to(dtype=self.dtype, device=self.device)
        self.triangles = torch.from_numpy(mesh.Elems()['Trias'][:, :-1]).to(dtype=torch.long, device=self.device)
        self.fibers = torch.from_numpy(mesh.Fibres()).to(dtype=self.dtype, device=self.device)

        if self.rcm is not None:
            self.rcm.calculate_rcm_order(self.vertices, self.triangles)

    def add_material_property(self, material_config):
        self.material_config = material_config

    def set_stimulus_region(self, path="/Users/bei/Project/FinitePDE/data/Case_1.vtx"):
        self.stimulus_region = load_stimulus_region(path)  # (2168,)

    def add_stimulus(self, stim_config):
        S = torch.zeros(size=(self.n_nodes,), device=self.device, dtype=torch.bool)
        S[self.stimulus_region] = True
        if self.rcm is not None:
            S = self.rcm.reorder(S)

        stimulus = Stimulus(stim_config)
        stimulus.set_stimregion(S)

        self.stimuli.append(stimulus)

    def assemble(self):
        if self.rcm is not None:
            rcm_vertices = self.rcm.reorder(self.vertices)
            rcm_triangles = self.rcm.map(self.triangles)
        else:
            rcm_vertices = self.vertices
            rcm_triangles = self.triangles

        # region_ids = domain.Elems()['Trias'][:, -1]
        sigma_l = self.material_config["diffusl"]
        sigma_t = self.material_config["diffust"]
        sigma = sigma_t * torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0).expand(self.fibers.shape[0], 3, 3)
        sigma += (sigma_l - sigma_t) * self.fibers.unsqueeze(2) @ self.fibers.unsqueeze(1)

        matrices = Matrices3DSurface(vertices=rcm_vertices, triangles=rcm_triangles, device=self.device, dtype=self.dtype)
        K, M = matrices.assemble_matrices(sigma)
        self.K = K.to(device=self.device, dtype=self.dtype)
        self.M = M.to(device=self.device, dtype=self.dtype)
        
        #######################################################
        # C = - self.K * (self.dt / 2)
        # A = self.M - C
        A = self.M + self.K * self.dt

        self.pcd = Preconditioner()
        self.pcd.create_Jocobi(A)

        self.A = A.to_sparse_csr()
        # self.C = C.to_sparse_csr()
        #######################################################

    def solve(self, a_tol, r_tol, max_iter, plot_interval=10, verbose=True):
        u = self.ionic_model.initialize(self.n_nodes)
        if self.rcm is not None:
            u = self.rcm.reorder(u)

        cg = ConjugateGradient(self.pcd)
        cg.initialize(x=u)

        ts_per_frame = int(plot_interval / self.dt)
        ctime = 0
        visualization = VTK3DSurface(self.vertices.cpu(), self.triangles.cpu())
        solving_time = time.time()

        n_total_iter = 0
        for n in range(1, self.nt + 1):
            ctime += 0.5*self.dt
            #######################################################
            du = self.ionic_model.differentiate(u)
            
            b = u + 0.5*self.dt * du
            for stimulus in self.stimuli:
                I0 = stimulus.stimApp(ctime)
                b += 0.5*self.dt * I0
            # b is now u at time n+1/2
            # b = self.M @ b + self.C @ u
            b = self.M @ b
            #######################################################
            u, n_iter = cg.solve(self.A, b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
            n_total_iter += n_iter
            ctime += 0.5*self.dt
            du = self.ionic_model.differentiate(u)
            b = u + 0.5*self.dt * du
            for stimulus in self.stimuli:
                I0 = stimulus.stimApp(ctime)
                b += 0.5*self.dt * I0
            u=b

            if n_iter == max_iter:
                raise Exception(f"The solution did not converge at {n}th timestep")
            # if verbose:
            #     print(f"{n} / {self.nt + 1}: {n_iter}; {round(time.time() - solving_time, 2)}")

            # if n % ts_per_frame == 0:
            #     visualization.save_frame(color_values=self.rcm.inverse(u).cpu().numpy() if self.rcm is not None else u.cpu().numpy(),
            #                              frame_path=f"./vtk_files_{self.n_nodes}_{self.rcm is not None}/frame_{n}.vtk")

        print(f"Ran {n_total_iter} iterations in {round(time.time() - solving_time, 2)} seconds")





if __name__ == "__main__":
    from ionic import CourtemancheRamirezNattel
    import torch
    from pathlib import Path

    simulation_time = 3000
    dt = 0.01
    stim_config = {'tstart': 0.0,
                'nstim': 3,
                'period': 800,
                'duration': 2.0,
                'intensity': 100.0,
                'name': 'S1'}
    material_config = {"vg": 0.1,
                    "diffusl": 0.4 * 1000 * 1000,
                    "diffust": 0.4 * 1000 * 1000,
                    "tin": 0.15,
                    "tout": 1.5,
                    "topen": 105.0,
                    "tclose": 185.0}


    device = torch.device(f"cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # dtype = torch.float32
    dtype = torch.float64

    home_directory = Path.home()
    ionic_model = CourtemancheRamirezNattel(dt=dt/2.0, device=device, dtype=dtype)
    print(ionic_model.default_parameters())
    ionic_model.reset_parameters(ACh=0.000001, Cao=1.8, Cm=100.)
    simulator = AtriumSimulatorCourtemanche(ionic_model, T=simulation_time, dt=dt, apply_rcm=True, device=device, dtype=dtype)
    simulator.load_mesh(path=f"{home_directory}/Data/atrium/Case_10/Case_10")
    simulator.add_material_property(material_config)
    simulator.set_stimulus_region(path=f"{home_directory}/Data/atrium/Case_10/Case_10.vtx")
    simulator.add_stimulus(stim_config)
    simulator.assemble()
    simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=1000, plot_interval=10, verbose=True)


