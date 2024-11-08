import torch
from preconditioner import Preconditioner
import time

class ConjugateGradient:
    def __init__(self, preconditioner: Preconditioner):
        self.preconditioner = preconditioner
        self.x = None
        self.x_prev = None

    def initialize(self, x): # Initial guess
        self.x = x.clone()
        self.x_prev = x.clone()

    def solve(self, A, b, a_tol=1e-6, r_tol=1e-6, max_iter=100):
        total_iter = 0

        x_clone = self.x.clone()
        self.x.mul_(2).sub_(self.x_prev)
        self.x_prev = x_clone

        r = b - A @ self.x # Initial residual
        z = self.preconditioner.apply(r)  # Preconditioned residual
        p = z.clone()  # Initial search direction

        r_norm = torch.linalg.vector_norm(r)
        # z_norm = torch.linalg.norm(z)
        for i in range(max_iter):

            total_iter += 1
            
            Ap = A @ p  # Matrix-vector product A*p
            rz_scala = torch.dot(r, z)
            alpha = rz_scala / torch.dot(p, Ap)  # Step size

            self.x.add_(alpha * p)

            r.sub_(alpha * Ap)
            
            r_new_norm = torch.linalg.vector_norm(r)
            # z_new_norm = torch.linalg.vector_norm(r)

            if r_new_norm < a_tol or (r_new_norm / r_norm) < r_tol:
                break
            # if z_new_norm < a_tol or (z_new_norm / z_norm) < r_tol:
            #     break

            z = self.preconditioner.apply(r)

            beta = torch.dot(r, z) / rz_scala
            print(beta)
            # p = z_new + beta * p
            p.mul_(beta).add_(z)

        return self.x.clone(), total_iter



