import sys
import os
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
from mesh.materialproperties import MaterialProperties
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

        self.stimulus_region = None
        self.stimuli = []

    def load_mesh(self, path="/Users/bei/Project/FinitePDE/data/Case_1"):
        mesh = Triangulation()
        mesh.readMesh(path)

        self.region_ids = torch.from_numpy(mesh.Elems()['Tetras'][:, -1]).to(dtype=torch.long, device=self.device)
        self.point_region_ids = mesh.point_region_ids()
        self.n_nodes = self.point_region_ids.shape[0]

        self.vertices = torch.from_numpy(mesh.Pts()).to(dtype=self.dtype, device=self.device)
        self.triangles = torch.from_numpy(mesh.Elems()['Tetras'][:, :-1]).to(dtype=torch.long, device=self.device)
        self.fibers = torch.from_numpy(mesh.Fibres()).to(dtype=self.dtype, device=self.device)

        if self.rcm is not None:
            self.rcm.calculate_rcm_order(self.vertices, self.triangles)

    def add_material_property(self, material_config):
        self.material_config = material_config

        # material = MaterialProperties()
        # nodal_properties = material.nodal_property_names()
        #
        # for npr in nodal_properties:
        #     npr_type = material.nodal_property_type(npr)
        #     attribute_value = self.ionic_model.get_attribute(npr)
        #
        #     if attribute_value is not None:
        #         if npr_type == "uniform":
        #             values = material.NodalProperty(npr, -1, -1)
        #         else:
        #             values = torch.full(size=(self.n_nodes, 1), fill_value=attribute_value)
        #             for point_id, region_id in enumerate(self.point_region_ids):
        #                 values[point_id] = material.NodalProperty(npr, point_id, region_id)
        #         self.ionic_model.set_attribute(npr, values)

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
        A = self.M + self.K * self.dt

        self.pcd = Preconditioner()
        self.pcd.create_Jocobi(A)
        self.A = A.to_sparse_csr()

    def solve(self, a_tol, r_tol, max_iter, plot_interval=10, verbose=True):
        u = self.ionic_model.initialize(self.n_nodes, self.dt)
        if self.rcm is not None:
            u = self.rcm.reorder(u)

        cg = ConjugateGradient(self.pcd)
        cg.initialize(x=u)

        ts_per_frame = int(plot_interval / self.dt)
        visualization = VTK3D(self.vertices.cpu().numpy(), self.triangles.cpu().numpy())
        visualization.save_frame(
            color_values=self.rcm.inverse(u).cpu().numpy() if self.rcm is not None else u.cpu().numpy(),
            frame_path=f"./vtk_files_{self.n_nodes}_{self.rcm is not None}/frame_{0}.vtk")

        ctime = 0
        solving_time = time.time()
        for n in range(1, self.nt + 1):
            ctime += self.dt
            du = self.ionic_model.differentiate(u)
            b = u + self.dt * du
            for stimulus in self.stimuli:
                I0 = stimulus.stimApp(ctime)
                b += self.dt * I0
            b = self.M @ b

            u, total_iter = cg.solve(self.A, b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)

            if total_iter == max_iter:
                raise Exception(f"The solution did not converge at {n} iteration")
            if verbose:
                print(f"{n} / {self.nt + 1}: {total_iter}; {round(time.time() - solving_time, 2)}")

            if n % ts_per_frame == 0:
                visualization.save_frame(color_values=self.rcm.inverse(u).cpu().numpy() if self.rcm is not None else u.cpu().numpy(),
                                         frame_path=f"./vtk_files_{self.n_nodes}_{self.rcm is not None}/frame_{n}.vtk")

        print(f"Solved in {round(time.time() - solving_time, 2)} seconds")







