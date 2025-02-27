import torch
from core.preconditioner import Preconditioner


@torch.jit.script
class ConjugateGradient:
    def __init__(self, preconditioner: Preconditioner):
        self.preconditioner = preconditioner
        self.x = torch.empty(0)
        self.x_prev = torch.empty(0)

        self.r = torch.empty(0)
        self.z = torch.empty(0)
        self.p = torch.empty(0)
        self.Ap = torch.empty(0)

    def initialize(self, x): # Initial guess
        self.x = x.clone()
        self.x_prev = x.clone()

        self.r = torch.empty_like(self.x)
        self.z = torch.empty_like(self.x)
        self.p = torch.empty_like(self.x)
        self.Ap = torch.empty_like(self.x)

    def initial_guess(self):
        x_clone = self.x.clone()
        self.x.mul_(2).sub_(self.x_prev)
        self.x_prev.copy_(x_clone)

    def solve(self, A, b, a_tol: float=1e-6, r_tol: float = 1e-6, max_iter: int = 100):
        n_iter: int = 0

        # self.initial_guess()
        
        self.r.copy_(b - A @ self.x) 
        self.z.copy_(self.preconditioner.apply(self.r))  
        self.p.copy_(self.z) 
        b_norm = torch.linalg.vector_norm(self.preconditioner.apply(b))

        for i in range(max_iter):
            self.Ap.copy_(A @ self.p)  
            rz_scala = torch.dot(self.r, self.z)
            alpha = rz_scala / torch.dot(self.p, self.Ap)  

            self.x.add_(alpha * self.p)
            self.r.sub_(alpha * self.Ap)
            
            self.z.copy_(self.preconditioner.apply(self.r))
            z_new_norm = torch.linalg.vector_norm(self.z)
            
            if z_new_norm.item() < a_tol or (z_new_norm.item() / b_norm.item()) < r_tol:
                break

            beta = torch.dot(self.r, self.z) / rz_scala
            
            self.p.mul_(beta).add_(self.z)
            n_iter += 1

        return self.x.clone(), n_iter