from cellml.ten_tusscher_model_2006_IK1Ko_endo_units import computeRates, sizeAlgebraic, createLegends, initConsts
import torch

class TenTusscher:
    def __init__(self, device, dtype, npt):
        self.H = None
        self.dt = None
        self.device = device
        self.dtype = dtype

        (_, _, _, legend_constants) = createLegends()

        init_states, init_constants = initConsts()
        self.states = torch.tensor(init_states, device=device, dtype=dtype).repeat(npt, 1)
        self.constants = torch.tensor(init_constants, device=device, dtype=dtype)

        for legend_constant, init_constant in zip(legend_constants, init_constants):
            constant_name = legend_constant.split()[0]
            if not constant_name.startswith("stim"):
                setattr(self, constant_name, init_constant)

    def initialize(self, dt):
        U = self.states[:, 0].clone()
        self.H = self.states[:, 1:].clone()
        self.dt = dt

        return U

    def differentiate(self, U):
        self.states[:, 0] = U
        # self.states[:, 1:] = self.H
        # print(self.states.numpy().tolist())

        rates = self.compute_rates(states=self.states, constants=self.constants)
        # print(rates)
        dU = rates[:, 0]
        dH = rates[:, 1:]

        self.H += self.dt * dH

        # print(dU)
        return dU

    def compute_rates(self, states, constants):
        rates = torch.zeros_like(states)
        algebraic = torch.zeros((states.shape[0], sizeAlgebraic), device=self.device, dtype=self.dtype)


        return rates

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    # tt = TenTusscher(None, None)
    # U = tt.initialize(dt=0.01, npt=1)
    # tt.differentiate(U)
    URES=[]
    tt = TenTusscher("cpu", torch.float64, npt=1)
    dt=0.01
    U = tt.initialize(dt=dt)
    Istim=100

    for jj in range(int(6000)):
        if(jj>100):
            Istim=0
        dU = tt.differentiate(U)
        U += dt*(dU+Istim)
        # print(U)
        # print(dU)
        # tt.states[0] = U
        if jj%100==0:
            URES.append(U.item())
    print(np.array(URES))
    plt.plot(np.array(URES))
    plt.show()

# 0.2557799338534495
# -0.0005820993186588484
# -0.1797664021371198
# -0.32453295203186705
# -0.45517011520109435
# -0.5803122656948434

