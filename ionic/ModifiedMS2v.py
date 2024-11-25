import torch


class ModifiedMS2v:
    def __init__(self, device=None, dtype=torch.float64):
        self.tau_in = 0.1
        self.tau_out = 9.0
        self.tau_open = 100.0
        self.tau_close = 120.0
        self.u_gate = 0.13
        self.u_crit = 0.13
        self._vmin = -80.0
        self._vmax = 20.0
        self._DV  = self._vmax-self._vmin


        self.H = None
        self.dt = None
        self.device = device
        self.dtype = dtype


    def to_dimensionless(self,U):
        """ to_dimensionless(U) rescales U to its dimensionless values (range [0,1])
        """
        return(U-self._vmin)/self._DV

    def derivative_to_dimensional(self,dU):
        """ derivative_to_dimensional(U) rescales the derivative of U (dU) to dimensional values
        """
        return(self._DV*dU)


    def initialize(self, n_nodes, dt):
        self.H = torch.full(size=(n_nodes,), fill_value=1.0, device=self.device, dtype=self.dtype)
        self.dt = dt

        u = torch.full(size=(n_nodes,), fill_value=self._vmin, device=self.device, dtype=self.dtype)
        return u

    def differentiate(self, U):
        Uad   = self.to_dimensionless(U)
        J_in  =  -1.0 * self.H * Uad * (Uad-self._u_crit) * (1.0-Uad)/self._tau_in
        J_out =  (1.0-self.H)*Uad/self._tau_out
        dU    = - self.derivative_to_dimensional(J_in +J_out)
        dH = torch.where(U > self.u_gate, -self.H / self.tau_close, (1 - self.H) / self.tau_open)
        self.H += self.dt * dH
        return dU

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
