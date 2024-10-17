import torch
from preconditioner import Preconditioner


class ConjugateGradient:
    def __init__(self, preconditioner: Preconditioner, device, dtype):
        self.preconditioner = preconditioner
        self.device = device
        self.dtype = dtype

    def solve(self, A, b, a_tol=1e-6, r_tol=1e-6, max_iter=100):
        x = torch.zeros_like(b, device=self.device, dtype=self.dtype)  # Initial guess (zero vector)
        r = b - A @ x  # Initial residual
        z = self.preconditioner.apply(r)  # Preconditioned residual
        p = z  # Initial search direction

        r_norm = torch.linalg.vector_norm(r)

        for i in range(max_iter):
            Ap = A @ p  # Matrix-vector product A*p
            rz_scala = torch.dot(r, z)
            alpha = rz_scala / torch.dot(p, Ap)  # Step size

            # Update the solution
            x = x + alpha * p

            # Update residual
            r_new = r - alpha * Ap

            r_new_norm = torch.linalg.vector_norm(r_new)
            # print(r_new_norm / r_norm)

            if r_new_norm < a_tol or (r_new_norm / r_norm) < r_tol:
                print(f"Converged in {i} iterations", r_new_norm < a_tol, (r_new_norm / r_norm) < r_tol)
                break

            # Apply the preconditioner iteratively
            z_new = self.preconditioner.apply(r_new)

            # Compute beta for the next search direction
            beta = torch.dot(r_new, z_new) / rz_scala

            # Update the search direction
            p = z_new + beta * p

            # Update residuals and preconditioned residuals for next iteration
            r = r_new
            z = z_new

        else:
            print(f"Did not converge in {i} iterations", r_new_norm < a_tol, (r_new_norm / r_norm) < r_tol)

        return x


