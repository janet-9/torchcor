import torch
import numpy as np
from assemble import assemble_matrices
from preconditioner import Preconditioner
from sovler import ConjugateGradient
from utils import Visualization
from matplotlib import tri
from boundary import apply_dirichlet_boundary_conditions


# Step 1: Define problem parameters
L = 1000  # Length of domain in x and y directions
T0 = 100
Nx, Ny = L, L  # Number of grid points in x and y
alpha = 6  # Thermal diffusivity
# h = 0.5206164
# print(h ** 2 / (2*alpha))
dt = 0.0125  # Time step size
nt = 1500  # Number of time steps

device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
dtype = torch.float64
print(device)

# Step 2: Generate grid (structured triangular mesh)
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)
triangulation = tri.Triangulation(X.flatten(), Y.flatten())

# Step 3: Initial condition
u0 = torch.zeros((L * L,)).to(device=device, dtype=torch.float64)
u0[30 * L: 30 * L + 3 * L] = T0
u = u0

K, M = assemble_matrices(triangulation, alpha)
K = K.to(device=device, dtype=dtype)
M = M.to(device=device, dtype=dtype)

M_dt = M * (1 / dt)
A = M_dt + K

# apply initial condition for A
print("applying boundary condition for A")
dirichlet_boundary_nodes = torch.arange(30 * L, 30 * L + 3 * L, device=device)
boundary_values = torch.ones_like(dirichlet_boundary_nodes, device=device, dtype=dtype) * T0

A = apply_dirichlet_boundary_conditions(A, dirichlet_boundary_nodes)

pcd = Preconditioner()
pcd.create_Jocobi(A)
cg = ConjugateGradient(pcd)

U = torch.zeros((nt, L, L)).to(device=device, dtype=dtype)
U[0, :, :] = u0.reshape((L, L))

print("solving")
for n in range(1, nt):
    b = torch.sparse.mm(M_dt, u.unsqueeze(1)).squeeze(1)

    b[dirichlet_boundary_nodes] = boundary_values  # apply initial condition for b

    u, converge = cg.solve(A, b, a_tol=1e-5, r_tol=1e-5, max_iter=100)

    if not converge:
        print(f"The solution did not converge at {n} iteration")
    else:
        print(f"{n} / {nt}")
    U[n, :, :] = u.reshape((L, L))

print("saving gif file")
visualization = Visualization(U, triangulation, dt)
visualization.save_gif("FEM - 2D Heat Equation - PCG - Sparse.gif")
