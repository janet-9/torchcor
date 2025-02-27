import torch
from petsc4py import PETSc


residuals = []

def monitor(ksp, it, anorm):
    """Custom monitor function to track residual norms."""
    print(f"Iteration {it}: Residual Norm = {anorm}")
    residuals.append((it, anorm))  # Store iteration and residual

coo_data = torch.load('A.pt', weights_only=True)
b_torch = torch.load('b.pt', weights_only=True)

row_indices = coo_data['row'].tolist()
col_indices = coo_data['col'].tolist()
data_values = coo_data['data'].tolist()
print(len(row_indices), len(col_indices), len(data_values))

A = PETSc.Mat().createAIJ(size=(637480, 637480))
for row, col, data in zip(row_indices, col_indices, data_values):
    A.setValue(row, col, data)
A.assemble()

b = PETSc.Vec().createSeq(len(b_torch))
b.setArray(b_torch.numpy())

# x = PETSc.Vec().createSeq(len(b_torch))
x = A.createVecRight()  # Creates a compatible vector for A's columns
x.set(-86.2)

ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setInitialGuessNonzero(True)
ksp.setMonitor(monitor)
opts = PETSc.Options()

opts['ksp_type'] = 'cg'
opts['ksp_max_it'] = 100
opts['pc_type'] = 'jacobi'
opts['pc_jacobi_type'] = 'diagonal'
opts['ksp_atol'] = 1e-5
opts['ksp_rtol'] = 1e-5
opts['ksp_monitor'] = None
opts['ksp_monitor_true_residual'] = None

ksp.setFromOptions()

print((ksp.getPC().getType()))

ksp.solve(b, x)

num_iterations = ksp.getIterationNumber()
print(num_iterations)

print(ksp.view())

print(residuals)