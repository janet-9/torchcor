import sys
import os
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from core.assemble import Matrices3D
from core.preconditioner import Preconditioner
from core.solver import ConjugateGradient
from core.visualize import VTK3D
from core.reorder import RCM as RCM
import time
from mesh.triangulation import Triangulation
from mesh.stimulus import Stimulus
from monodomain.tools import load_stimulus_region


class VentricleSimulator:
    def __init__(self, ionic_model, T, dt, apply_rcm, device=None, dtype=None):
        self.device = torch.device(device) if device is not None else "cuda:0" \
            if torch.cuda.is_available() else "cpu"
        self.dtype = dtype if dtype is not None else torch.float64

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
        self.region_ids = None

        self.material_config = None

        self.K = None
        self.M = None
        self.A = None
        self.Chi = 140 
        self.Cm = 0.01  
        self.theta = 0.5

        self.stimulus_region = None
        self.stimuli = []

    def load_mesh(self, path="/Users/bei/Project/FinitePDE/data/Case_1"):
        mesh = Triangulation()
        mesh.readMesh(path)

        self.region_ids = torch.from_numpy(mesh.Elems()['Tetras'][:, -1]).to(dtype=torch.long, device=self.device)
        # raise Exception(np.unique(mesh.Elems()['Tetras'][:, -1]))
        self.point_region_ids = mesh.point_region_ids()
        self.n_nodes = self.point_region_ids.shape[0]

        self.vertices = torch.from_numpy(mesh.Pts()).to(dtype=self.dtype, device=self.device) / 1000
        self.triangles = torch.from_numpy(mesh.Elems()['Tetras'][:, :-1]).to(dtype=torch.long, device=self.device)
        self.fibers = torch.from_numpy(mesh.Fibres()).to(dtype=self.dtype, device=self.device)

        if self.rcm is not None:
            self.rcm.calculate_rcm_order(self.vertices, self.triangles)

    def add_material_property(self, material_config):
        self.material_config = material_config

    def add_stimulus(self, stim_region_path, stim_config):
        self.stimulus_region = load_stimulus_region(stim_region_path)

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

        sigma_l = torch.zeros_like(self.region_ids, dtype=self.dtype)
        sigma_t = torch.zeros_like(self.region_ids, dtype=self.dtype)

        # Iterate through the mapping and replace values using torch.where
        for region_id, value in self.material_config['diffusl'].items():
            sigma_l = torch.where(self.region_ids == region_id,
                                  torch.tensor(value, device=self.device, dtype=self.dtype),
                                  sigma_l)

        for region_id, value in self.material_config['diffust'].items():
            sigma_t = torch.where(self.region_ids == region_id,
                                  torch.tensor(value, device=self.device, dtype=self.dtype),
                                  sigma_t)
        sigma_l = sigma_l.view(sigma_l.shape[0], 1, 1)
        sigma_t = sigma_t.view(sigma_t.shape[0], 1, 1)
        sigma = sigma_t * torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0).expand(self.fibers.shape[0], 3, 3)
        sigma += (sigma_l - sigma_t) * self.fibers.unsqueeze(2) @ self.fibers.unsqueeze(1)

        matrices = Matrices3D(vertices=rcm_vertices, tetrahedrons=rcm_triangles, device=self.device, dtype=self.dtype)
        K, M = matrices.assemble_matrices(sigma)

        self.K = K.to(device=self.device, dtype=self.dtype)
        self.M = M.to(device=self.device, dtype=self.dtype)
        A = self.M * self.Cm * self.Chi + self.K * self.dt * self.theta

        self.pcd = Preconditioner()
        self.pcd.create_Jocobi(A)
        self.A = A.to_sparse_csr()

    def solve(self, a_tol, r_tol, max_iter, plot_interval=10, verbose=True):
        u = self.ionic_model.initialize(self.n_nodes)
        if self.rcm is not None:
            u = self.rcm.reorder(u)

        cg = ConjugateGradient(self.pcd)
        cg.initialize(x=u)

        ts_per_frame = int(plot_interval / self.dt)
        visualization = VTK3D(self.vertices.cpu().numpy(), self.triangles.cpu().numpy())

        ctime = 0
        solving_time = time.time()
        n_total_iter = 0
        for n in range(1, self.nt + 1):
            ctime += self.dt
            du = self.ionic_model.differentiate(u) / 100

            b = u * self.Cm + self.dt * du
            for stimulus in self.stimuli:
                I0 = stimulus.stimApp(ctime) / self.Chi
                b += self.dt * I0
            
            b = self.Chi * self.M @ b 
            b -= (1 - self.theta) * self.dt * self.K @ u
            
            u, n_iter = cg.solve(self.A, b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
            n_total_iter += n_iter

            if n_iter == max_iter:
                raise Exception(f"The solution did not converge at {n}th timestep")
            if verbose:
                print(f"{n} / {self.nt + 1}: {n_iter}; {round(time.time() - solving_time, 2)}")
            
            if n % ts_per_frame == 0:
                visualization.save_frame(color_values=self.rcm.inverse(u).cpu().numpy() if self.rcm is not None else u.cpu().numpy(),
                                         frame_path=f"./vtk_files_{self.n_nodes}_{self.rcm is not None}/frame_{n}.vtk")
        
        print(f"Ran {n_total_iter} iterations in {round(time.time() - solving_time, 2)} seconds")



if __name__ == "__main__":
    from monodomain import VentricleSimulator
    from ionic import TenTusscherPanfilov
    import torch
    from pathlib import Path

    simulation_time = 1000
    dt = 0.01
    stim_LV_sf = {'tstart': 0.0,
                  'nstim': 1,
                  'period': 800,
                  'duration': 1.0,
                  'intensity': 100.0,
                  'name': 'LV_sf'}
    stim_LV_pf = {'tstart': 0.0,
                  'nstim': 1,
                  'period': 800,
                  'duration': 1.0,
                  'intensity': 100.0,
                  'name': 'LV_pf'}
    stim_LV_af = {'tstart': 0.0,
                  'nstim': 1,
                  'period': 800,
                  'duration': 1.0,
                  'intensity': 100.0,
                  'name': 'LV_af'}
    stim_RV_sf = {'tstart': 5.0,
                  'nstim': 1,
                  'period': 800,
                  'duration': 1.0,
                  'intensity': 100.0,
                  'name': 'RV_sf'}
    stim_RV_mod = {'tstart': 5.0,
                   'nstim': 1,
                   'period': 800,
                   'duration': 1.0,
                   'intensity': 100.0,
                   'name': 'RV_mod'}

    il = 0.5272
    it = 0.2076
    el = 1.0732 
    et = 0.4227
    l_34_35 = il * el * (1 / (il + el))
    t_34_35 = it * et * (1 / (it + et))

    il = 0.9074 
    it = 0.3332
    el = 0.9074
    et = 0.3332
    l_44_45_46 = il * el * (1 / (il + el))
    t_44_45_46 = it * et * (1 / (it + et))

    material_config = {"diffusl": {34: l_34_35,
                                   35: l_34_35,
                                   44: l_44_45_46,
                                   45: l_44_45_46,
                                   46: l_44_45_46},
                       "diffust": {34: t_34_35,
                                   35: t_34_35,
                                   44: t_44_45_46,
                                   45: t_44_45_46,
                                   46: t_44_45_46}}


    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    home_dir = Path.home()

    ionic_model = TenTusscherPanfilov(cell_type="ENDO", dt=dt, device=device)
    simulator = VentricleSimulator(ionic_model, T=simulation_time, dt=dt, apply_rcm=True, device=device)
    simulator.load_mesh(path=f"{home_dir}/Data/ventricle/biv")
    simulator.add_material_property(material_config)

    simulator.add_stimulus(f"{home_dir}/Data/ventricle/LV_sf.vtx", stim_LV_sf)
    simulator.add_stimulus(f"{home_dir}/Data/ventricle/LV_pf.vtx", stim_LV_pf)
    simulator.add_stimulus(f"{home_dir}/Data/ventricle/LV_af.vtx", stim_LV_af)
    simulator.add_stimulus(f"{home_dir}/Data/ventricle/RV_sf.vtx", stim_RV_sf)
    simulator.add_stimulus(f"{home_dir}/Data/ventricle/RV_mod.vtx", stim_RV_mod)
    simulator.assemble()
    simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=1000, plot_interval=10, verbose=True)

