import torch
from preconditioner import Preconditioner


class ConjugateGradient:
    def __init__(self, preconditioner: Preconditioner):
        self.preconditioner = preconditioner

    def solve(self, A, b, a_tol=1e-6, r_tol=1e-6, max_iter=100):
        device, dtype = A.device, A.dtype
        total_iter = 0

        x = torch.zeros_like(b, device=device, dtype=dtype)  # Initial guess (zero vector)
        r = b - A @ x  # Initial residual
        z = self.preconditioner.apply(r)  # Preconditioned residual
        p = z  # Initial search direction

        r_norm = torch.linalg.vector_norm(r)

        for i in range(max_iter):
            total_iter += 1
            Ap = A @ p  # Matrix-vector product A*p
            rz_scala = torch.dot(r, z)
            alpha = rz_scala / torch.dot(p, Ap)  # Step size

            x = x + alpha * p

            r_new = r - alpha * Ap

            r_new_norm = torch.linalg.vector_norm(r_new)
            # print(r_new_norm / r_norm)

            if r_new_norm < a_tol or (r_new_norm / r_norm) < r_tol:
                # print(f"Converged in {i} iterations")
                break

            z_new = self.preconditioner.apply(r_new)

            beta = torch.dot(r_new, z_new) / rz_scala

            p = z_new + beta * p

            r = r_new
            z = z_new

        return x, total_iter


