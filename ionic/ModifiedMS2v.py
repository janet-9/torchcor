import torch


class ModifiedMS2v:
    def __init__(self, device=None, dtype=torch.float64):
        self.tau_in = 0.1
        self.tau_out = 9.0
        self.tau_open = 100.0
        self.tau_close = 120.0
        self.u_gate = 0.13
        self.u_crit = 0.13

        self.H = None
        self.dt = None
        self.device = device
        self.dtype = dtype

    def initialize(self, n_nodes, dt):
        self.H = torch.full(size=(n_nodes,), fill_value=1.0, device=self.device, dtype=self.dtype)
        self.dt = dt

    def differentiate(self, U):
        J_in = -1.0 * self.H * U * (U - self.u_crit) * (1 - U) / self.tau_in
        J_out = (1 - self.H) * U / self.tau_out
        dU = - (J_in + J_out)

        dH = torch.where(U > self.u_gate, -self.H / self.tau_close, (1 - self.H) / self.tau_open)
        self.H += self.dt * dH

        return dU

    def set_attribute(self, name, value):
        setattr(self, name, value)

    def get_attribute(self, name: str):
        return getattr(self, name, None)


