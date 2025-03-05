import torch


class ModifiedMS2v:
    def __init__(self, dt, device, dtype):
        self.tau_in = 0.1
        self.tau_out = 9.0
        self.tau_open = 100.0
        self.tau_close = 120.0
        self.u_gate = 0.13
        self.u_crit = 0.13

        self.H = None
        self.dt = dt
        self.device = device
        self.dtype = dtype

    def construct_tables(self):
        pass

    def initialize(self, npt):
        self.H = torch.full(size=(npt,), fill_value=1.0, device=self.device, dtype=self.dtype)
        return self.H

    def differentiate(self, U):
        J_in = -1.0 * self.H * U * (U - self.u_crit) * (1 - U) / self.tau_in
        J_out = (1 - self.H) * U / self.tau_out
        dU = - (J_in + J_out)

        dH = torch.where(U > self.u_gate, -self.H / self.tau_close, (1 - self.H) / self.tau_open)
        self.H += self.dt * dH

        return dU / 100

    def set_attribute(self, name, value):
        setattr(self, name, value)

    def get_attribute(self, name: str):
        return getattr(self, name, None)
    


class ModifiedMS2vRL(ModifiedMS2v):
    def __init__(self, device=None, dtype=torch.float64):
        super().__init__(device, dtype)

    def differentiate(self, U):
        # Compute ionic currents
        Uad   = self.to_dimensionless(U)
        J_in  =  -1.0 * self.H * Uad * (Uad-self._u_crit) * (1.0-Uad)/self._tau_in
        J_out =  (1.0-self.H)*Uad/self._tau_out
        dU    = - self.derivative_to_dimensional(J_in +J_out)

        # Rush–Larsen update for H
        H_inf = torch.where(U > self.u_gate, 0.0, 1.0)  # Steady-state gating variable
        tau_H = torch.where(U > self.u_gate, self.tau_close, self.tau_open)  # Time constant
        self.H = H_inf + (self.H - H_inf) * torch.exp(-self.dt / tau_H)

        return dU
