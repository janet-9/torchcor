import torch
from torchfinite.preconditioner import Preconditioner
import time

class ConjugateGradient:
    def __init__(self, preconditioner: Preconditioner):
        self.preconditioner = preconditioner
        self.x = None

    def initialize(self, x): # Initial guess
        self.x = x  

    def solve(self, A, b, a_tol=1e-6, r_tol=1e-6, max_iter=100):
        total_iter = 0

        
        r = b - A @ self.x # Initial residual
        
        
        z = self.preconditioner.apply(r)  # Preconditioned residual
        
        p = z.clone()  # Initial search direction

        r_norm = torch.linalg.vector_norm(r)
        for i in range(1):
            total_iter += 1
            
            ap_time = time.time()
            Ap = torch.sparse.mm(A, p.view(-1, 1)).view(-1)  # Matrix-vector product A*p
            print(f"ap time {Ap.shape} { time.time() - ap_time}")
            rz_scala = torch.dot(r, z)
            alpha = rz_scala / torch.dot(p, Ap)  # Step size
            
            self.x.add_(alpha * p)

            r_new = r - alpha * Ap

            
            r_new_norm = torch.linalg.vector_norm(r_new)
            

            if r_new_norm < a_tol: #  or (r_new_norm / r_norm) < r_tol
                # print(f"Converged in {i} iterations")
                break

            z_new = self.preconditioner.apply(r_new)

            beta = torch.dot(r_new, z_new) / rz_scala

            # p = z_new + beta * p
            p.mul_(beta).add_(z_new)
            r = r_new
            z = z_new

        return self.x, total_iter



