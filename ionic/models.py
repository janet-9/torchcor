import numpy as np
import torch
from numpy import *
from ten_tusscher_model_2006_IK1Ko_endo_units import sizeAlgebraic, createLegends, initConsts

class ModifiedMS2v:
    def __init__(self, device, dtype):
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

    def initialize(self, dt, npt):
        self.H = torch.full(size=(npt,), fill_value=1.0, device=self.device, dtype=self.dtype)
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


class TenTusscher(ModifiedMS2v):
    def __init__(self, device, dtype):
        self.H = None
        self.dt = None
        self.device = device
        self.dtype = dtype

        (_, _, _, legend_constants) = createLegends()

        init_states, init_constants = initConsts()
        self.states = init_states
        self.constants = init_constants

        for legend_constant, init_constant in zip(legend_constants, init_constants):
            constant_name = legend_constant.split()[0]
            if not constant_name.startswith("stim"):
                setattr(self, constant_name, init_constant)

    def initialize(self, dt, npt):
        U = torch.full(size=(npt,), fill_value=self.states[0], device=self.device, dtype=self.dtype)
        self.H = torch.tensor(self.states[1:], device=self.device, dtype=self.dtype).repeat(npt, 1)
        self.dt = dt

        return U

    def differentiate(self, U):
        rates = self.computeRates(states=torch.cat((U, self.H), dim=1), constants=self.constants)

        dU = rates[0]
        dH = rates[1:]

        self.H += self.dt * dH
        return dU

    def computeRates(self, states, constants):
        rates = torch.zeros_like(states)
        algebraic = torch.zeros((states.shape[0], sizeAlgebraic), device=self.device, dtype=self.dtype)

        algebraic[:, 7] = 1.0 / (1.0 + torch.exp((states[:, 0] + 20.0) / 7.0))
        algebraic[:, 20] = (
                1102.5 * torch.exp(-((states[:, 0] + 27.0) ** 2) / 225.0)
                + 200.0 / (1.0 + torch.exp((13.0 - states[:, 0]) / 10.0))
                + 180.0 / (1.0 + torch.exp((states[:, 0] + 30.0) / 10.0))
                + 20.0
        )
        rates[:, 12] = (algebraic[:, 7] - states[:, 12]) / algebraic[:, 20]


if __name__ == "__main__":
   URES=[]
   tt = TenTusscher(None, None)
   dt=0.01
   U = tt.initialize(dt=dt, npt=100)
   Istim=100

   for jj in range(int(800//dt)):
       if(jj>100):
           Istim=0
       dU = tt.differentiate(U)
       U += dt*(dU+Istim)
       tt.states[0]=U
       if jj%100==0:
        URES.append(U)
import matplotlib.pyplot as plt
plt.plot(np.array(URES))
plt.show()
