import torch
import numpy as np
from assemble import Matrices
from preconditioner import Preconditioner
from sovler import ConjugateGradient
from utils import Visualization
from matplotlib import tri
from boundary import apply_dirichlet_boundary_conditions
import time

# Step 1: Define problem parameters
L = 1  # Length of domain in x and y directions
Nx = 100
Ny = 100  # Number of grid points in x and y
T0 = 100
alpha = 0.1  # Thermal diffusivity
# h = 0.5206164
# print(h ** 2 / (2 * alpha))
dt = 0.00125  # Time step size
nt = 1000  # Number of time steps
ts_per_frame = 1
max_iter = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
print(device)

# Step 2: Generate grid (structured triangular mesh)
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)

# Y[1:Nx-1, 1: Ny-1] += np.random.rand(Nx-2, Ny-2) * 0.5
# X[1:Nx-1, 1: Ny-1] -= np.random.rand(Nx-2, Ny-2) * 0.5

triangulation = tri.Triangulation(X.flatten(), Y.flatten())

# Step 3: Initial condition
u0 = torch.zeros((Nx * Ny,)).to(device=device, dtype=dtype)
u0[20 * Nx: 20 * Nx + Ny] = T0
u = u0

start = time.time()
print("assembling matrices")
matrices = Matrices(device=device, dtype=dtype)
K, M = matrices.assemble_matrices(triangulation, alpha)
K = K.to(device=device, dtype=dtype)
M = M.to(device=device, dtype=dtype)
print(f"assembled in: {time.time() - start} seconds")

# print(K.to_dense().numpy())

M_dt = M * (1 / dt)
A = M_dt + K

# apply initial condition for A
print("applying boundary condition for A")
dirichlet_boundary_nodes = torch.arange(20 * Nx, 20 * Nx + Ny, device=device)
boundary_values = torch.ones_like(dirichlet_boundary_nodes, device=device, dtype=dtype) * T0

A = apply_dirichlet_boundary_conditions(A, dirichlet_boundary_nodes)

pcd = Preconditioner()
pcd.create_Jocobi(A)
cg = ConjugateGradient(pcd)
cg.initialize(x=u)

frames = u0.reshape((1, Nx, Ny))
start = time.time()
print("solving")
for n in range(1, nt):
    b = M_dt @ u
    b[dirichlet_boundary_nodes] = boundary_values  # apply initial condition for b

    u, total_iter = cg.solve(A, b, a_tol=1e-5, r_tol=1e-5, max_iter=max_iter)
    if total_iter == max_iter:
        print(f"The solution did not converge at {n} iteration")
    else:
        print(f"{n} / {nt}: {total_iter}")

    if n % ts_per_frame == 0:
        frames = torch.cat((frames, u.reshape((1, Nx, Ny))))

print(f"solved in: {time.time() - start} seconds")

print("saving gif file")
visualization = Visualization(frames, triangulation, dt, ts_per_frame)
visualization.save_gif("FEM - 2D Heat Equation - PCG - Sparse.gif")




