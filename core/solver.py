import torch
from core.preconditioner import Preconditioner
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
        self.x_prev.copy_(x_clone)

        r = b - A @ self.x # Initial residual
        z = self.preconditioner.apply(r)  # Preconditioned residual
        p = z.clone()  # Initial search direction

        z_norm = torch.linalg.norm(z)
        for i in range(max_iter):

            total_iter += 1
            
            Ap = A @ p  # Matrix-vector product A*p
            rz_scala = torch.dot(r, z)
            alpha = rz_scala / torch.dot(p, Ap)  # Step size

            self.x.add_(alpha * p)

            r.sub_(alpha * Ap)

            z_new_norm = torch.linalg.vector_norm(z)

            if z_new_norm < a_tol or (z_new_norm / z_norm) < r_tol:
                break

            z = self.preconditioner.apply(r)

            beta = torch.dot(r, z) / rz_scala

            p.mul_(beta).add_(z)

        return self.x.clone(), total_iter



class BiCGStab:
    def __init__(self, preconditioner):
        self.preconditioner = preconditioner
        self.x = None
        self.x_prev = None

    def initialize(self, x):  # Initial guess
        self.x = x.clone()
        self.x_prev = x.clone()

    def solve(self, A, b, a_tol=1e-6, r_tol=1e-6, max_iter=100):
        total_iter = 0

        x_clone = self.x.clone()
        self.x.mul_(2).sub_(self.x_prev)
        self.x_prev.copy_(x_clone)
        
        # Initial residuals
        r = b - A @ self.x  # Initial residual
        r_tilde = r.clone()  # Fixed vector (typically, initial residual)
        
        # Scalars for the algorithm
        rho_old = alpha = omega = 1
        v = torch.zeros_like(self.x)
        p = torch.zeros_like(self.x)

        # Norm of the initial residual for relative tolerance checks
        r_norm_init = torch.linalg.norm(r)
        
        for i in range(max_iter):
            total_iter += 1

            # Calculate rho
            rho_new = torch.dot(r_tilde, r)
            if rho_new.abs() < 1e-10:  # Avoid division by zero
                break

            # Calculate beta
            if i > 0:
                beta = (rho_new / rho_old) * (alpha / omega)
                p = r + beta * (p - omega * v)
            else:
                p.copy_(r)

            # Preconditioning step for p
            p_hat = self.preconditioner.apply(p)
            v = A @ p_hat
            alpha = rho_new / torch.dot(r_tilde, v)

            # Update the intermediate solution s
            s = r - alpha * v
            s_norm = torch.linalg.norm(s)
            if s_norm < a_tol:
                self.x.add_(alpha * p_hat)
                break

            # Preconditioning step for s
            s_hat = self.preconditioner.apply(s)
            t = A @ s_hat
            omega = torch.dot(t, s) / torch.dot(t, t)

            # Update solution x and residual r
            self.x.add_(alpha * p_hat + omega * s_hat)
            r = s - omega * t
            r_norm = torch.linalg.norm(r)

            # Convergence check
            if r_norm < a_tol or (r_norm / r_norm_init) < r_tol:
                break

            # Update rho for next iteration
            rho_old = rho_new

        return self.x.clone(), total_iter