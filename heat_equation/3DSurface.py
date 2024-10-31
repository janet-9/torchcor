import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from assemble import Matrices3D
from preconditioner import Preconditioner
from solver import ConjugateGradient
from boundary import apply_dirichlet_boundary_conditions
import time
from reorder import RCM
import numpy as np
from utils import set_logger
from visualize import Visualization3DSurface
from assemble import Matrices3DSurface
from scipy.spatial import Delaunay
import argparse

parser = argparse.ArgumentParser(description="A simple example of argparse.")
parser.add_argument("-n", '--n_grid', type=int, default=100)
parser.add_argument("-c", '--cuda', type=int, default=0)
parser.add_argument('--no_rcm', action='store_false')
parser.add_argument('--vtk', action='store_true')
args = parser.parse_args()

# Step 1: Define problem parameters
T0 = 100
L = 1  # Length of domain in x and y directions
Nx = args.n_grid
Ny = args.n_grid  # Number of grid points in x and y
alpha = 0.001  # Thermal diffusivity
dt = 0.0125  # Time step size
nt = 1000  # Number of time steps
ts_per_frame = 10
max_iter = 100
apply_rcm = args.no_rcm
save_frames = args.vtk

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
if device.type != "cpu":
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
print(f"Using {device}")


start = time.time()
print("Constructing mesh: ", end="")
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)
# Y = 0.5 * Y
# Z = math.sqrt(3) * Y
# Z = X + Y
# Z = X ** 2 + Y
Z = np.sqrt(X**2 + Y**2)
print(f"done in: {time.time() - start} seconds")

points = np.vstack([X.flatten(), Y.flatten()]).T
vertices = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T 
n_vertices = len(vertices)
triangles = Delaunay(points).simplices
n_triangles = len(triangles)
print(f"Vertices (Nodes): {n_vertices}, Triangles: {n_triangles}", end=" ")


# Step 3: RCM ordering and matrix assembly
print("Assembling matrices: ", end="")
start = time.time()
rcm = RCM(device=device, dtype=dtype)
rcm_vertices, rcm_triangles = rcm.calculate_rcm_order(vertices, triangles)
matrices = Matrices3DSurface(rcm_vertices, rcm_triangles, device=device, dtype=dtype)
K, M = matrices.assemble_matrices(alpha)
assemble_matrix_time = time.time() - start
print(f"done in: {assemble_matrix_time} seconds")


M_dt = M * (1 / dt)
A = M_dt + K

# apply initial condition for A
boundary_nodes = torch.arange(80 * Nx, 80 * Nx + Ny, device=device)
if apply_rcm:
    boundary_nodes = rcm.apply(boundary_nodes)
boundary_values = torch.ones_like(boundary_nodes, device=device, dtype=dtype) * T0

u0 = torch.zeros((Nx * Ny,), device=device, dtype=dtype)
u0[boundary_nodes] = boundary_values
u = u0

A = apply_dirichlet_boundary_conditions(A, boundary_nodes)

pcd = Preconditioner()
pcd.create_Jocobi(A)
cg = ConjugateGradient(pcd)
cg.initialize(x=u)

# LU, pivots = torch.linalg.lu_factor(A.to_dense())

frames = rcm.inverse(u0).reshape((1, Nx, Ny))
start = time.time()
for n in range(nt):
    b = M_dt @ u
    b[boundary_nodes] = boundary_values  # Apply initial condition for b

    u, total_iter = cg.solve(A, b, a_tol=1e-5, r_tol=1e-5, max_iter=max_iter)
    if total_iter == max_iter:
        print(f"The solution did not converge at {n} iteration")
    else:
        print(f"{n} / {nt}: {total_iter}")

    if n % ts_per_frame == 0 and save_frames:
        frames = torch.cat((frames, rcm.inverse(u).reshape((1, Nx, Ny))))
solving_time = time.time() - start

max_memory_used = torch.cuda.max_memory_allocated(device=device.index) / 1024 ** 3
logger = set_logger("3DSurface.log")
logger.info(f"Solved {n_vertices} nodes ({Nx}), {n_triangles} triangles for {nt} timesteps in {round(solving_time, 2)} seconds; "
            f"Assemble: {round(assemble_matrix_time, 2)}; "
            f"RCM:{apply_rcm}; "
            f"Memory Usage: {round(max_memory_used, 4)} GB")
print(f"Solved {n_vertices} nodes ({Nx}) for {nt} timesteps in {solving_time} seconds; RCM:{apply_rcm}")


if save_frames:
    print("Saving frames: ", end="")
    visualization = Visualization3DSurface(frames, vertices, triangles, dt, ts_per_frame)
    visualization.save_gif(f"./(Z = np.sqrt(X**2 + Y**2)) 3D surface L={L}, alpha={alpha}, dx=dy={L/Nx}.gif")
    print("Done")




