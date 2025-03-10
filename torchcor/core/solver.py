import torch
from torchcor.core.preconditioner import Preconditioner


@torch.jit.script
class ConjugateGradient:
    def __init__(self, preconditioner: Preconditioner, A: torch.Tensor, dtype: torch.dtype = torch.float64) -> None:
        self.preconditioner = preconditioner
        self.dtype = dtype
        self.A = A.to_sparse_csr()

        self.x = torch.empty(0)
        self.x_prev = torch.empty(0)
        
        self.r = torch.empty(0)
        self.z = torch.empty(0)
        self.p = torch.empty(0)
        self.Ap = torch.empty(0)

        self.linear_guess = False


    def initialize(self, x: torch.Tensor, linear_guess: bool = True) -> None:
        self.A = self.A.to(self.dtype)
        self.x = x.to(self.dtype).clone()
        self.x_prev = x.clone()

        self.r = torch.empty_like(self.x)
        self.z = torch.empty_like(self.x)
        self.p = torch.empty_like(self.x)
        self.Ap = torch.empty_like(self.x)

        self.linear_guess = linear_guess

    def initial_guess(self) -> None:
        x_clone = self.x.clone()
        self.x.mul_(2).sub_(self.x_prev)
        self.x_prev.copy_(x_clone)

    def solve(self, 
              b: torch.Tensor, 
              a_tol: float = 1e-6, 
              r_tol: float = 1e-6, 
              max_iter: int = 100) -> tuple[torch.Tensor, int]:
        b_dtype = b.dtype
        b = b.to(self.dtype)
        n_iter: int = 0
        
        if self.linear_guess:
            self.initial_guess()
        
        self.r.copy_(b - self.A @ self.x) 
        self.z.copy_(self.preconditioner.apply(self.r))  
        self.p.copy_(self.z) 
        b_norm = torch.linalg.vector_norm(self.preconditioner.apply(b))

        for _ in range(max_iter):
            self.Ap.copy_(self.A @ self.p)  
            rz_scala = torch.dot(self.r, self.z)
            alpha = rz_scala / torch.dot(self.p, self.Ap)  

            self.x.add_(alpha * self.p)
            self.r.sub_(alpha * self.Ap)
            
            self.z.copy_(self.preconditioner.apply(self.r))
            z_new_norm = torch.linalg.vector_norm(self.z)
            # raise Exception(z_new_norm.item() / b_norm.item(), r_tol)
            if z_new_norm.item() < a_tol or (z_new_norm.item() / b_norm.item()) < r_tol:
                break

            beta = torch.dot(self.r, self.z) / rz_scala
            
            self.p.mul_(beta).add_(self.z)
            n_iter += 1

        return self.x.to(b_dtype), n_iter