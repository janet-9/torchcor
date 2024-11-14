import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from core.assemble import Matrices3D
from core.preconditioner import Preconditioner
from core.solver import ConjugateGradient
from core.boundary import apply_dirichlet_boundary_conditions
import time
from core.reorder import RCM
import pygmsh
from core.utils import set_logger
from core.visualize import VTK3D
import argparse

parser = argparse.ArgumentParser(description="A simple example of argparse.")
parser.add_argument("-s", '--mesh_size', type=float, default=0.1)
parser.add_argument("-nt", '--n_timesteps', type=int, default=1000)
parser.add_argument("-c", '--cuda', type=int, default=0)
parser.add_argument('--no-rcm', action='store_false')
parser.add_argument('--vtk', action='store_true')
args = parser.parse_args()

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
if device.type != "cpu":
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats()
print(f"Using {device}")

# Step 1: Define problem parameters
T0 = 100  # Initial temperature on the boundary
sigma = torch.eye(3, device=device, dtype=dtype) * 0.001  # Thermal diffusivity
dt = 0.0015  # Time step size
nt = args.n_timesteps  # Number of time steps
ts_per_frame = 100  # Frames per output time step
max_iter = 100  # Max CG iterations
apply_rcm = args.no_rcm
mesh_size = args.mesh_size
save_frames = args.vtk



start = time.time()
print("Constructing mesh: ", end="")
with pygmsh.geo.Geometry() as geom:
    # Define a box with specific corner points (x0, y0, z0) and (x1, y1, z1)
    cube = geom.add_box(x0=0, x1=1, y0=0, y1=1, z0=0, z1=1, mesh_size=mesh_size)
    mesh = geom.generate_mesh()
print(f"done in: {time.time() - start} seconds")

vertices = mesh.points   # (235, 3)
n_vertices = len(vertices)
tetrahedrons = mesh.cells_dict["tetra"]   # (718, 4)
n_tetrahedrons = len(tetrahedrons)
print(f"Vertices (Nodes): {n_vertices}, tetrahedrons {n_tetrahedrons}", end=" ")

# Step 3: RCM ordering and matrix assembly
print("Assembling matrices: ", end="")
start = time.time()

rcm = RCM(device=device, dtype=dtype)
if apply_rcm:
    rcm_vertices, rcm_tetrahedrons = rcm.calculate_rcm_order(vertices, tetrahedrons)
else:
    rcm_vertices = torch.from_numpy(vertices).to(dtype=dtype, device=device)
    rcm_tetrahedrons = torch.from_numpy(tetrahedrons).to(dtype=torch.long, device=device)

matrices = Matrices3D(rcm_vertices,
                      rcm_tetrahedrons,
                      device=device, dtype=dtype)
K, M = matrices.assemble_matrices(sigma)
assemble_matrix_time = time.time() - start
print(f"done in: {assemble_matrix_time} seconds")

M_dt = M * (1 / dt)
A = M_dt + K

# Step 4: Apply Dirichlet boundary conditions
boundary_nodes = torch.tensor([i for i, (x, y, z) in enumerate(vertices) if x == 0], device=device, dtype=torch.long)
if apply_rcm:
    boundary_nodes = rcm.apply(boundary_nodes)
boundary_values = torch.ones_like(boundary_nodes, device=device, dtype=dtype) * T0

u0 = torch.zeros((vertices.shape[0],), device=device, dtype=dtype)
u0[boundary_nodes] = boundary_values
u = u0

A = apply_dirichlet_boundary_conditions(A, boundary_nodes)

# Step 5: Preconditioner and CG Solver Setup
pcd = Preconditioner()
pcd.create_Jocobi(A)
A = A.to_sparse_csr()

cg = ConjugateGradient(pcd)
cg.initialize(x=u)


# Step 6: Time-stepping solution
start = time.time()
frames = [(0, u)] if save_frames else []
for n in range(nt):
    b = M_dt @ u
    b[boundary_nodes] = boundary_values  # Apply initial condition for b

    u, total_iter = cg.solve(A, b, a_tol=1e-5, r_tol=1e-5, max_iter=max_iter)
    if total_iter == max_iter:
        print(f"The solution did not converge at {n} iteration")
    else:
        print(f"{n} / {nt}: {total_iter}")

    if n % ts_per_frame == 0 and save_frames:
        frames.append((n, u))
solving_time = time.time() - start

max_memory_used = torch.cuda.max_memory_allocated(device=device.index) / 1024 ** 3
logger = set_logger("./logs/3D.log")
logger.info(f"Solved {n_vertices} nodes ({mesh_size}), {n_tetrahedrons} tetrahedrons for {nt} timesteps in {round(solving_time, 2)} seconds; "
            f"Assemble: {round(assemble_matrix_time, 2)}; "
            f"RCM:{apply_rcm}; "
            f"Memory Usage: {round(max_memory_used, 4)} GB")
print(f"Solved {n_vertices} nodes ({mesh_size}) for {nt} timesteps in {solving_time} seconds; RCM:{apply_rcm}")


# print(frames[0][1] == frames[1][1])
if save_frames:
    visualization = VTK3D(vertices, tetrahedrons)
    print("Saving frames: ", end="")
    for n, u in frames:
        visualization.save_frame(color_values=rcm.inverse(u).cpu().numpy() if apply_rcm else u.cpu().numpy(),
                                 frame_path=f"./vtk_files_{len(vertices)}_{apply_rcm}/frame_{n}.vtk")
    print("Done")


