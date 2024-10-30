import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from assemble import Matrices3D
from preconditioner import Preconditioner
from solver import ConjugateGradient
from boundary import apply_dirichlet_boundary_conditions
import time
from reorder import RCM
import pygmsh
import meshio
from utils import Visualization3D

# Step 1: Define problem parameters
T0 = 100  # Initial temperature on the boundary

alpha = 0.01  # Thermal diffusivity
dt = 0.0125  # Time step size
nt = 1000  # Number of time steps
ts_per_frame = 10  # Frames per output time step
max_iter = 100  # Max CG iterations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
print(f"Using {device}")

print("Constructing mesh: ", end="")
start = time.time()

with pygmsh.geo.Geometry() as geom:
    # Define a box with specific corner points (x0, y0, z0) and (x1, y1, z1)
    cube = geom.add_box(x0=0, x1=1, y0=0, y1=1, z0=0, z1=1, mesh_size=0.005)
    mesh = geom.generate_mesh()

    vertices = mesh.points   # (235, 3)
    n_vertices = len(vertices)
    tetrahedrons = mesh.cells_dict["tetra"]   # (718, 4)

    meshio.write(f"./3D_Mesh/{n_vertices}_nodes.vtk", mesh)
    print(f"Vertices (Nodes): {n_vertices}", end="")
print(f"done in: {time.time() - start} seconds")


# Step 3: RCM ordering and matrix assembly
print("Assembling matrices: ", end="")
start = time.time()

rcm = RCM(device=device, dtype=dtype)
rcm_vertices, rcm_tetrahedrons = rcm.calculate_rcm_order(vertices, tetrahedrons)
matrices = Matrices3D(rcm_vertices,
                      rcm_tetrahedrons,
                      device=device, dtype=dtype)
K, M = matrices.assemble_matrices(alpha)
print(f"done in: {time.time() - start} seconds")

M_dt = M * (1 / dt)
A = M_dt + K

# Step 4: Apply Dirichlet boundary conditions
boundary_nodes = torch.tensor([i for i, (x, y, z) in enumerate(vertices) if x == 0], device=device, dtype=torch.long)
boundary_nodes = rcm.apply(boundary_nodes)
boundary_values = torch.ones_like(boundary_nodes, device=device, dtype=dtype) * T0

u0 = torch.zeros((vertices.shape[0],), device=device, dtype=dtype)
u0[boundary_nodes] = boundary_values
u = u0

A = apply_dirichlet_boundary_conditions(A, boundary_nodes)

# Step 5: Preconditioner and CG Solver Setup
pcd = Preconditioner()
pcd.create_Jocobi(A)
cg = ConjugateGradient(pcd)
cg.initialize(x=u)

visualization = Visualization3D(vertices, tetrahedrons)
# Step 6: Time-stepping solution
start = time.time()
frames = [(0, u)]
for n in range(nt):
    b = M_dt @ u
    b[boundary_nodes] = boundary_values  # Apply initial condition for b

    u, total_iter = cg.solve(A, b, a_tol=1e-5, r_tol=1e-5, max_iter=max_iter)
    if total_iter == max_iter:
        print(f"The solution did not converge at {n} iteration")
    else:
        print(f"{n} / {nt}: {total_iter}")

    if n % ts_per_frame == 0:
        frames.append((n, u))
print(f"Solved {n_vertices} nodes in: {time.time() - start} seconds")

print("Saving frames: ", end="")
for n, u in frames:
    visualization.save_frame(color_values=rcm.inverse(u).cpu().numpy(),
                             frame_path=f"./vtk_files_{len(vertices)}/frame_{n}.vtk")
print("Done")


