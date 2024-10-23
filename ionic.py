import torch


class ModifiedMS2v:
    def __init__(self):
        self.tau_in = 0.1
        self.tau_out = 9.0
        self.tau_open = 100.0
        self.tau_close = 120.0
        self.u_gate = 0.13
        self.u_crit = 0.13

    def differentiate(self, U, H):
        J_in = -1.0 * H * U (U - self.u_crit) * (1 - U) / self.tau_in
        J_out = (1 - H) * U / self.tau_out
        dU = - (J_in + J_out)
        dH = torch.where(U > self.u_gate, -H / self.tau_close, (1 - H) / self.tau_open)
        return dU, dH

    def set_attribute(self, name, value):
        if name in self.__dict__.keys():
            setattr(self, name, value)

    def get_attribute(self, name: str):
        return getattr(self, name, None)
