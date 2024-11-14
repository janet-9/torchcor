import sys
import os


# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import numpy as np
from core.assemble import Matrices2D
from core.preconditioner import Preconditioner
from core.solver import ConjugateGradient
from core.visualize import Visualization2D
from core.boundary import apply_dirichlet_boundary_conditions
import time
from scipy.spatial import Delaunay
from core.reorder import RCM

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
print(device)

# Step 1: Define problem parameters
L = 1  # Length of domain in x and y directions
Nx = 100
Ny = 100  # Number of grid points in x and y
T0 = 100

sigma = torch.tensor([[0.001 * 0.5, -0.001 * 0.5],
                           [-0.001 * 0.5, 0.001 * 0.5]], device=device, dtype=dtype)  # Thermal diffusivity
# h = 0.5206164
# print(h ** 2 / (2 * alpha))
dt = 0.0125  # Time step size
nt = 1000  # Number of time steps
ts_per_frame = 10
max_iter = 100

# Step 2: Generate grid (structured triangular mesh)
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

# Y[1:Nx-1, 1: Ny-1] += np.random.rand(Nx-2, Ny-2) * 0.5
# X[1:Nx-1, 1: Ny-1] -= np.random.rand(Nx-2, Ny-2) * 0.5

# Flatten the X, Y, Z arrays for input to Delaunay
vertices = np.vstack([X.flatten(), Y.flatten()]).T
print(vertices.shape)
triangles = Delaunay(vertices).simplices


print(f"Vertices: {len(vertices)}, Nodes: {len(triangles)}")

start = time.time()
print("assembling matrices")
rcm = RCM(device=device, dtype=dtype)
rcm_vertices, rcm_triangles = rcm.calculate_rcm_order(vertices, triangles)

matrices = Matrices2D(rcm_vertices, rcm_triangles, device=device, dtype=dtype)
K, M = matrices.assemble_matrices(sigma)
print(f"assembled in: {time.time() - start} seconds")


M_dt = M * (1 / dt)
A = M_dt + K

# apply initial condition for A
print("applying boundary condition for A")
dirichlet_boundary_nodes = torch.arange(30 * Nx, 30 * Nx + Ny)
dirichlet_boundary_nodes = rcm.apply(dirichlet_boundary_nodes).to(device)
boundary_values = torch.ones_like(dirichlet_boundary_nodes, device=device, dtype=dtype) * T0

u0 = torch.zeros((Nx * Ny,), device=device, dtype=dtype)
u0[dirichlet_boundary_nodes] = boundary_values
u = u0


A = apply_dirichlet_boundary_conditions(A, dirichlet_boundary_nodes)

pcd = Preconditioner()
pcd.create_Jocobi(A)
A = A.to_sparse_csc()

cg = ConjugateGradient(pcd)
cg.initialize(x=u)

# LU, pivots = torch.linalg.lu_factor(A.to_dense())

frames = rcm.inverse(u0).reshape((1, Nx, Ny))

# frames = u0.reshape((1, Nx, Ny))
start = time.time()
print("solving")
for n in range(0, nt):
    b = M_dt @ u
    b[dirichlet_boundary_nodes] = boundary_values  # apply initial condition for b

    # u = torch.linalg.lu_solve(LU, pivots, b.unsqueeze(1)).squeeze()

    u, total_iter = cg.solve(A, b, a_tol=1e-5, r_tol=1e-5, max_iter=max_iter)
    if total_iter == max_iter:
        print(f"The solution did not converge at {n} iteration")
    else:
        print(f"{n} / {nt}: {total_iter}")

    if n % ts_per_frame == 0:
        frames = torch.cat((frames, rcm.inverse(u).reshape((1, Nx, Ny))))
        # frames = torch.cat((frames, u.reshape((1, Nx, Ny))))

print(f"solved in: {time.time() - start} seconds")

print("saving gif file")
visualization = Visualization2D(frames, vertices, triangles, dt, ts_per_frame)
visualization.save_gif("./2D Heat Equation.gif")




