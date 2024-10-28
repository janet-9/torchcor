import math
import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import numpy as np
from assemble import Matrices3DSurface
from preconditioner import Preconditioner
from sovler import ConjugateGradient
from utils import Visualization3DSurface, Visualization
from boundary import apply_dirichlet_boundary_conditions
import time
from scipy.spatial import Delaunay

# Step 1: Define problem parameters
L = 1  # Length of domain in x and y directions
Nx = 200
Ny = 200  # Number of grid points in x and y
T0 = 100
alpha = 0.001  # Thermal diffusivity
dt = 0.0125  # Time step size
nt = 2000  # Number of time steps
ts_per_frame = 10
max_iter = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
print(device)


# Step 2: Generate grid (structured triangular mesh)
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)
# Y = 0.5 * Y
# Z = math.sqrt(3) * Y
# Z = X + Y
# Z = X ** 2 + Y
Z = np.sqrt(X**2 + Y**2)

points = np.vstack([X.flatten(), Y.flatten()]).T
vertices = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T 
triangles = Delaunay(points).simplices
triangles.sort(axis=1)


print(vertices.shape, triangles.shape)
print(f"Vertices (Nodes): {len(vertices)}, Triangles: {len(triangles)}")

# Step 3: Initial condition
u0 = torch.zeros((Nx * Ny,)).to(device=device, dtype=dtype)
u0[80 * Nx: 80 * Nx + Ny] = T0
u = u0

start = time.time()
print("assembling matrices")
matrices = Matrices3DSurface(vertices, triangles, device=device, dtype=dtype)
K, M = matrices.assemble_matrices(alpha)
print(f"assembled in: {time.time() - start} seconds")

# print(K.to_dense().numpy())

M_dt = M * (1 / dt)
A = M_dt + K

# apply initial condition for A
print("applying boundary condition for A")
dirichlet_boundary_nodes = torch.arange(80 * Nx, 80 * Nx + Ny, device=device)
boundary_values = torch.ones_like(dirichlet_boundary_nodes, device=device, dtype=dtype) * T0

A = apply_dirichlet_boundary_conditions(A, dirichlet_boundary_nodes)

pcd = Preconditioner()
pcd.create_Jocobi(A)
cg = ConjugateGradient(pcd)
cg.initialize(x=u)

# LU, pivots = torch.linalg.lu_factor(A.to_dense())

frames = u0.reshape((1, Nx, Ny))
start = time.time()
print("solving")
for n in range(1, nt):
    b = M_dt @ u
    b[dirichlet_boundary_nodes] = boundary_values  # apply initial condition for b

    # u = torch.linalg.lu_solve(LU, pivots, b.unsqueeze(1)).squeeze()

    u, total_iter = cg.solve(A, b, a_tol=1e-5, r_tol=1e-5, max_iter=max_iter)
    if total_iter == max_iter:
        print(f"The solution did not converge at {n} iteration")
    else:
        print(f"{n} / {nt}: {total_iter}")

    if n % ts_per_frame == 0:
        frames = torch.cat((frames, u.reshape((1, Nx, Ny))))

print(f"solved in: {time.time() - start} seconds")

print("saving gif file")
print(frames.shape)
visualization = Visualization3DSurface(frames, vertices, triangles, dt, ts_per_frame)
visualization.save_gif(f"./(Z = np.sqrt(X**2 + Y**2)) 3D surface L={L}, alpha={alpha}, dx=dy={L/Nx}.gif")





