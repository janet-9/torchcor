import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import numpy as np
from assemble import Matrices3D
from preconditioner import Preconditioner
from solver import ConjugateGradient
from utils import Visualization
from boundary import apply_dirichlet_boundary_conditions
import time
from scipy.spatial import Delaunay
from reorder import RCM
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Step 1: Define problem parameters
L = 1  # Length of domain in x, y, z directions
Nx, Ny, Nz = 20, 20, 20  # Number of grid points in each direction
T0 = 100  # Initial temperature on the boundary

alpha = 0.001  # Thermal diffusivity
dt = 0.0125  # Time step size
nt = 1000  # Number of time steps
ts_per_frame = 10  # Frames per output time step
max_iter = 100  # Max CG iterations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
print(device)

# Step 2: Generate 3D grid (structured tetrahedral mesh)
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
z = np.linspace(0, L, Nz)
X, Y, Z = np.meshgrid(x, y, z)
vertices = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T

# Generate tetrahedral mesh using Delaunay triangulation
tetrahedrons = Delaunay(vertices).simplices
print(f"Vertices: {len(vertices)}, Tetrahedrons: {len(tetrahedrons)}")


# # Plot the mesh in 3D
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# # Plot a subset of tetrahedrons for clarity
# for tetra in tetrahedrons:  # Limit to 100 tetrahedrons for visibility
#     points = vertices[tetra]
#     # Create a Poly3DCollection for the faces of the tetrahedron
#     ax.add_collection3d(Poly3DCollection(
#         [points[[0, 1, 2]], points[[0, 1, 3]], points[[0, 2, 3]], points[[1, 2, 3]]],
#         color='skyblue', edgecolor='k', linewidths=0.5, alpha=0.2))
# # Set the limits and labels
# ax.set_xlim(0, L)
# ax.set_ylim(0, L)
# ax.set_zlim(0, L)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.title("3D Tetrahedral Mesh")

# plt.show()


# Step 3: RCM ordering and matrix assembly
start = time.time()
print("assembling matrices")
rcm = RCM()
rcm_vertices, rcm_tetrahedrons = vertices, tetrahedrons # rcm.calculate_rcm_order(vertices, tetrahedrons)


matrices = Matrices3D(torch.tensor(rcm_vertices, device=device, dtype=dtype), 
                      torch.tensor(rcm_tetrahedrons, device=device, dtype=torch.long),
                      device=device, dtype=dtype)
K, M = matrices.assemble_matrices(alpha)
print(f"assembled in: {time.time() - start} seconds")

M_dt = M * (1 / dt)
A = M_dt + K

# Step 4: Apply Dirichlet boundary conditions
print("applying boundary condition for A")
boundary_nodes = torch.tensor([i for i, v in enumerate(vertices) if v[0] == 0 or v[0] == L or 
                               v[1] == 0 or v[1] == L or v[2] == 0 or v[2] == L], device=device, dtype=torch.long)
boundary_values = torch.ones_like(boundary_nodes, device=device, dtype=dtype) * T0

u0 = torch.zeros((Nx * Ny * Nz,), device=device, dtype=dtype)
u0[boundary_nodes] = boundary_values
u = u0

A = apply_dirichlet_boundary_conditions(A, boundary_nodes)

# Step 5: Preconditioner and CG Solver Setup
pcd = Preconditioner()
pcd.create_Jocobi(A)
cg = ConjugateGradient(pcd)
cg.initialize(x=u)

# frames = rcm.inverse(u0).reshape((1, Nx, Ny, Nz))
frames = u0.reshape((1, Nx, Ny, Nz))

# Step 6: Time-stepping solution
start = time.time()
print("solving")
for n in range(nt):
    b = M_dt @ u
    b[boundary_nodes] = boundary_values  # Apply initial condition for b

    u, total_iter = cg.solve(A, b, a_tol=1e-5, r_tol=1e-5, max_iter=max_iter)
    if total_iter == max_iter:
        print(f"The solution did not converge at {n} iteration")
    else:
        print(f"{n} / {nt}: {total_iter}")

    if n % ts_per_frame == 0:
        # frames = torch.cat((frames, rcm.inverse(u).reshape((1, Nx, Ny, Nz))))
        frames = torch.cat((frames, u.reshape((1, Nx, Ny, Nz))))

print(f"solved in: {time.time() - start} seconds")

# Step 7: Visualization
# print("saving gif file")
# visualization = Visualization(frames, vertices, tetrahedrons, dt, ts_per_frame)
# visualization.save_gif("./3D_Heat_Equation.gif")
