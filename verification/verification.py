import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


import pygmsh
from TenTusscherPanfilov import TenTusscherPanfilov
import torch
from core.assemble import Matrices3D
from core.preconditioner import Preconditioner
from core.solver import ConjugateGradient
from core.visualize import VTK3D
from core.reorder import RCM as RCM
import numpy as np
import time


class Monodomain:
    def __init__(self, ionic_model, T, dt, apply_rcm, device=None, dtype=None):
        self.device = torch.device(device) if device is not None else "cuda:0" \
            if torch.cuda.is_available() else "cpu"
        self.dtype = dtype if dtype is not None else torch.float64

        self.T = T  # ms 
        self.dt = dt  # ms
        self.dx = None
        self.nt = int(T // dt)
        self.rcm = RCM(device=device, dtype=dtype) if apply_rcm else None

        self.ionic_model = ionic_model
        self.ionic_model.construct_tables()

        self.pcd = None

        self.point_region_ids = None
        self.n_nodes = None
        self.vertices = None
        self.tetrahedral = None
        self.fibers = None
        self.region_ids = None

        self.material_config = None

        
        self.Chi = 140  # mm
        self.Cm = 0.01  # uFmm
        self.K = None
        self.M = None
        self.A = None

        self.stimulus_region = None
        self.stimuli = []

    def load_mesh(self, dx):
        self.dx = dx

        length = 20  # mm
        width = 7    # mm
        height = 3   # mm

        # Create the cube with tethrahedrals
        with pygmsh.geo.Geometry() as geom:
            geom.add_box(
                x0=0, x1=length,  
                y0=0, y1=width,  
                z0=0, z1=height, 
                mesh_size=dx
            )
            mesh = geom.generate_mesh()

        self.vertices = torch.from_numpy(mesh.points).to(dtype=self.dtype, device=self.device)
        self.n_nodes = self.vertices.shape[0]
        self.tetrahedral = torch.from_numpy(mesh.cells_dict["tetra"]).to(dtype=torch.long, device=self.device)
        self.fibers = torch.tensor([1, 0, 0]).repeat(self.tetrahedral.shape[0], 1).to(dtype=self.dtype, device=self.device)

        corner_size = 1.5
        self.corner_indices = torch.where((self.vertices[:, 0] <= corner_size) &  # x <= 1.5
                                       (self.vertices[:, 1] <= corner_size) &  # y <= 1.5
                                       (self.vertices[:, 2] <= corner_size)    # z <= 1.5
                                )[0] 
        
        self.P1_index = torch.where((self.vertices[:, 0] == 0) &  
                                    (self.vertices[:, 1] == 0) &  
                                    (self.vertices[:, 2] == 0)    
                                )[0]
        self.P8_index = torch.where((self.vertices[:, 0] == 20) &  
                                    (self.vertices[:, 1] == 7) &  
                                    (self.vertices[:, 2] == 3)   
                                )[0]

        print(self.vertices.shape, self.tetrahedral.shape, self.fibers.shape)

        if self.rcm is not None:
            self.rcm.calculate_rcm_order(self.vertices, self.tetrahedral)

    def add_material_property(self, material_config):
        self.material_config = material_config

    def assemble(self):
        if self.rcm is not None:
            rcm_vertices = self.rcm.reorder(self.vertices)
            rcm_tetrahedral = self.rcm.map(self.tetrahedral)
        else:
            rcm_vertices = self.vertices
            rcm_tetrahedral = self.tetrahedral

        sigma_l = self.material_config['diffusl']
        sigma_t = self.material_config['diffust']

        sigma = sigma_t * torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0).expand(self.fibers.shape[0], 3, 3)
        sigma += (sigma_l - sigma_t) * self.fibers.unsqueeze(2) @ self.fibers.unsqueeze(1)

        matrices = Matrices3D(vertices=rcm_vertices, tetrahedrons=rcm_tetrahedral, device=self.device, dtype=self.dtype)
        K, M = matrices.assemble_matrices(sigma)

        self.K = K.to(device=self.device, dtype=self.dtype)
        self.M = M.to(device=self.device, dtype=self.dtype)
        A = self.M * self.Cm * self.Chi + self.K * self.dt

        self.pcd = Preconditioner()
        self.pcd.create_Jocobi(A)
        self.A = A.to_sparse_csr()

    def solve(self, a_tol, r_tol, max_iter, plot_interval=10, verbose=True):
        u = self.ionic_model.initialize(self.n_nodes)

        stimulus = torch.zeros_like(u)
        stimulus[self.corner_indices] = 50

        if self.rcm is not None:
            u = self.rcm.reorder(u)

        cg = ConjugateGradient(self.pcd)
        cg.initialize(x=u)

        ts_per_frame = int(plot_interval / self.dt)
        visualization = VTK3D(self.vertices.cpu().numpy(), self.tetrahedral.cpu().numpy())

        ctime = 0
        solving_time = time.time()
        n_total_iter = 0
        activation_time = torch.zeros_like(u)

        for n in range(1, self.nt + 1):
            ctime += self.dt

            du = self.ionic_model.differentiate(u)
            
            b = u * self.Cm + self.dt * du
            # b = u * self.Cm 

            # apply the stimulus for 2
            if ctime <= 2.0:
                b += self.dt * stimulus

            b = self.M @ b * self.Chi
            
            u, n_iter = cg.solve(self.A, b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
            n_total_iter += n_iter

            activation_time[(u > 0) & (activation_time == 0)] = ctime

            print(u[self.P1_index].item(), u[self.P8_index].item(), u.min().item(), u.max().item())
            if u[self.P8_index].item() > 0:
                break
            
            if n_iter == max_iter:
                raise Exception(f"The solution did not converge at {n}th timestep")
            if verbose:
                print(f"{round(ctime, 3)} / {self.T}: {n_iter}; {round(time.time() - solving_time, 2)}")
            
            if n % ts_per_frame == 0:
                visualization.save_frame(color_values=self.rcm.inverse(u).cpu().numpy() if self.rcm is not None else u.cpu().numpy(),
                                         frame_path=f"./{self.n_nodes}_{self.rcm is not None}_dt{self.dt}_dx{self.dx}/frame_{n}.vtk")
        
        print(f"Ran {n_total_iter} iterations in {round(time.time() - solving_time, 2)} seconds")

        # np.save(f"activation_time_dt{self.dt}_dx{self.dx}.npy", activation_time.cpu().numpy())
        visualization.save_frame(color_values=self.rcm.inverse(activation_time).cpu().numpy() if self.rcm is not None else activation_time.cpu().numpy(),
                                 frame_path=f"activation_time_dt{self.dt}_dx{self.dx}_laplace.vtk")

if __name__ == "__main__":
    dt = 0.005  # ms
    dx = 0.1    # mm

    device = torch.device(f"cuda:3" if torch.cuda.is_available() else "cpu")

    ionic_model = TenTusscherPanfilov(cell_type="EPI", dt=dt, device=device)
    ionic_model.V_init = -85.23
    ionic_model.Xr1_init = 0.00621
    ionic_model.Xr2_init = 0.4712
    ionic_model.Xs_init = 0.0095
    ionic_model.M_init = 0.00172
    ionic_model.H_init = 0.7444
    ionic_model.J_init = 0.7045
    ionic_model.d_init = 3.373e-5
    ionic_model.F_init = 0.7888
    ionic_model.F2_init = 0.9755
    ionic_model.FCaSS_init = 0.9953
    ionic_model.S_init = 0.999998
    ionic_model.R_init = 2.42e-8
    ionic_model.Cai_init = 0.000126
    ionic_model.CaSR_init = 3.64
    ionic_model.CaSS_init = 0.00036
    ionic_model.R__init = 0.9073
    ionic_model.Nai_init = 8.064
    ionic_model.Ki_init = 136.89

    il = 0.17 
    it = 0.019 
    el = 0.62 
    et = 0.2
    material_config = {"diffusl": il * el * (1 / (il + el)),
                       "diffust": it * et * (1 / (it + et))}

    simulator = Monodomain(ionic_model, 
                           T=150, 
                           dt=dt, 
                           apply_rcm=False, 
                           device=device)
    simulator.Chi = 140
    simulator.Cm = 0.01
    simulator.load_mesh(dx=dx)
    simulator.add_material_property(material_config)
    simulator.assemble()
    simulator.solve(a_tol=1e-5, 
                    r_tol=1e-5, 
                    max_iter=1000, 
                    plot_interval=dt * 10, 
                    verbose=True)