import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)


import torch
from core.assemble import Matrices2D
from core.preconditioner import Preconditioner
from core.solver import ConjugateGradient
from core.visualize import Visualization2D
from core.reorder import RCM as RCM
from scipy.spatial import Delaunay
import numpy as np
import time
from core.boundary import apply_dirichlet_boundary_conditions


class Monodomain:
    def __init__(self, ionic_model, T, dt, apply_rcm, device=None, dtype=None):
        self.device = torch.device(device) if device is not None else "cuda:0" \
            if torch.cuda.is_available() else "cpu"
        self.dtype = dtype if dtype is not None else torch.float64

        self.T = T  # ms 
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

        
        self.Chi = 1  # mm
        self.Cm = 1  # uFmm
        self.K = None
        self.M = None
        self.A = None

        self.stimulus_region = None
        self.stimuli = []

    def load_mesh(self):

        L = 1  # Length of domain in x and y directions
        self.Nx = 100
        self.Ny = 100  # Number of grid points in x and y

        x = np.linspace(0, L, self.Nx)
        y = np.linspace(0, L, self.Ny)
        X, Y = np.meshgrid(x, y)

        # Flatten the X, Y, Z arrays for input to Delaunay
        vertices = np.vstack([X.flatten(), Y.flatten()]).T
        self.triangles = torch.from_numpy(Delaunay(vertices).simplices).to(dtype=torch.long, device=device)
        self.vertices = torch.from_numpy(vertices).to(dtype=self.dtype, device=device)

        self.n_nodes = self.vertices.shape[0]
        
        if self.rcm is not None:
            self.rcm.calculate_rcm_order(self.vertices, self.triangles)

    def add_material_property(self, material_config):
        self.material_config = material_config

    def assemble(self):
        if self.rcm is not None:
            rcm_vertices = self.rcm.reorder(self.vertices)
            rcm_triangles = self.rcm.map(self.triangles)
        else:
            rcm_vertices = self.vertices
            rcm_triangles = self.triangles
        
        sigma = torch.tensor([[0.001 * 0.5, 0],
                              [0, 0.001 * 0.5]], device=self.device, dtype=self.dtype)

        matrices = Matrices2D(rcm_vertices, rcm_triangles, device=self.device, dtype=self.dtype)
        K, M = matrices.assemble_matrices(sigma)
        
        self.K = K.to(device=self.device, dtype=self.dtype)
        self.M = M.to(device=self.device, dtype=self.dtype)
        A = self.M * self.Cm * self.Chi + self.K * self.dt

        dirichlet_boundary_nodes = torch.arange(0, self.Nx).to(device)
        A = apply_dirichlet_boundary_conditions(A, dirichlet_boundary_nodes)

        self.pcd = Preconditioner()
        self.pcd.create_Jocobi(A)
        self.A = A.to_sparse_csr()

    def solve(self, a_tol, r_tol, max_iter, plot_interval=10, verbose=True):
        
        dirichlet_boundary_nodes = torch.arange(0, self.Nx).to(device)
        if self.rcm is not None:
            dirichlet_boundary_nodes = self.rcm.map(dirichlet_boundary_nodes).to(device)
        boundary_values = torch.ones_like(dirichlet_boundary_nodes, device=self.device, dtype=self.dtype) * 100

        u0 = torch.zeros((self.Nx * self.Ny,), device=self.device, dtype=self.dtype)
        u0[dirichlet_boundary_nodes] = boundary_values
        u = u0

        if self.rcm is not None:
            u = self.rcm.reorder(u)

        cg = ConjugateGradient(self.pcd)
        cg.initialize(x=u)

        ts_per_frame = int(plot_interval / self.dt)

        ctime = 0
        solving_time = time.time()
        n_total_iter = 0
        if self.rcm is not None:
            frames = self.rcm.inverse(u).reshape((1, self.Nx, self.Ny))
        else:
            frames = u.reshape((1, self.Nx, self.Ny))

        for n in range(1, self.nt + 1):
            ctime += self.dt

            b = u * self.Cm 
            b = self.M @ b * self.Chi
            b[dirichlet_boundary_nodes] = boundary_values
            
            u, n_iter = cg.solve(self.A, b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
            n_total_iter += n_iter
            
            if n_iter == max_iter:
                raise Exception(f"The solution did not converge at {n}th timestep")
            if verbose:
                print(f"{round(ctime, 3)} / {self.T}: {n_iter}; {round(time.time() - solving_time, 2)}")
            
            if n % ts_per_frame == 0:
                if self.rcm is not None:
                    frames = torch.cat((frames, self.rcm.inverse(u).reshape((1, self.Nx, self.Ny))))
                else:
                    frames = torch.cat((frames, u.reshape((1, self.Nx, self.Ny))))

        visualization = Visualization2D(frames, self.vertices, self.triangles, self.dt, ts_per_frame)
        visualization.save_gif("./Analytical solution.gif")

if __name__ == "__main__":
    dt = 0.0125  # ms

    device = torch.device(f"cuda:3" if torch.cuda.is_available() else "cpu")

    simulator = Monodomain(ionic_model=None, 
                           T=10, 
                           dt=dt, 
                           apply_rcm=False, 
                           device=device)
    simulator.load_mesh()
    simulator.add_material_property(material_config=None)
    simulator.assemble()
    simulator.solve(a_tol=1e-5, 
                    r_tol=1e-5, 
                    max_iter=1000, 
                    plot_interval=dt * 10, 
                    verbose=True)