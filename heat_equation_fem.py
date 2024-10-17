import torch
import numpy as np
from assemble import assemble_matrices
from preconditioner import Preconditioner
from sovler import ConjugateGradient
from utils import Visualization
from matplotlib import tri


# Step 1: Define problem parameters
L = 50  # Length of domain in x and y directions
T0 = 100
Nx, Ny = L, L  # Number of grid points in x and y
alpha = 6  # Thermal diffusivity
# h = 0.5206164
# print(h ** 2 / (2*alpha))
dt = 0.0125  # Time step size
nt = 3000  # Number of time steps

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
u0[:L] = 100


K, M = assemble_matrices(triangulation, alpha)
K = K.to(device=device, dtype=dtype)
M = M.to(device=device, dtype=dtype)

u = u0
M_dt = M * (1 / dt)
A = M_dt + K

# apply initial condition for A
print("applying initial condition for A")
sparse_indices = A._indices()
sparse_values = A._values()

mask = sparse_indices[0] >= L
new_indices = sparse_indices[:, mask]
new_values = sparse_values[mask]

identify_array = torch.arange(L, device=device)
identity_indices = torch.stack([identify_array, identify_array], dim=0).to(device=device, dtype=dtype)  # Diagonal indices
identity_values = torch.ones(L).to(device=device, dtype=dtype)  # Diagonal values are set to 1.0

final_indices = torch.cat([new_indices, identity_indices], dim=1)
final_values = torch.cat([new_values, identity_values])
A = torch.sparse_coo_tensor(final_indices, final_values, A.shape)

pcd = Preconditioner(device=device, dtype=dtype)
pcd.create_Jocobi(A)
cg = ConjugateGradient(pcd, device=device, dtype=dtype)

U = torch.zeros((nt, L, L)).to(device=device, dtype=dtype)
U[0, :, :] = u0.reshape((L, L))

for n in range(1, nt):
    print(f"{n}", end=" ")
    b = torch.sparse.mm(M_dt, u.unsqueeze(1)).to(torch.float64).squeeze(1)

    b[:L] = T0  # apply initial condition for b
    u = cg.solve(A, b, a_tol=1e-5, r_tol=1e-5, max_iter=100)

    U[n, :, :] = u.reshape((L, L))


visualization = Visualization(U, triangulation, dt)
visualization.save_gif("FEM - 2D Heat Equation - PCG - Sparse.gif")
