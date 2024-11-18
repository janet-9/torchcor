from cellml.ten_tusscher_model_2006_IK1Ko_endo_units import createLegends, initConsts, computeRates
import torch
import numpy as np

class TenTusscher:
    def __init__(self, device, dtype, npt):
        self.H = None
        self.dt = None
        self.device = device
        self.dtype = dtype

        (_, _, _, legend_constants) = createLegends()

        init_states, init_constants = initConsts()
        self.states = np.array(init_states)
        self.constants = init_constants

        for legend_constant, init_constant in zip(legend_constants, init_constants):
            constant_name = legend_constant.split()[0]
            if not constant_name.startswith("stim"):
                setattr(self, constant_name, init_constant)

    def initialize(self, dt):
        U = self.states[0]
        self.H = np.array(self.states[1:]).copy()
        self.dt = dt

        return U

    def differentiate(self, U):
        self.states[0] = U
        self.states[1:] = self.H

        rates = computeRates(None, states=self.states.tolist(), constants=self.constants)

        dU = rates[0]
        dH = rates[1:]

        self.H += self.dt * np.array(dH)

        return dU



if __name__ == "__main__":
    TEND   = 1000   # final time (in ms)
    dt     = 0.001  # time step
    dt_out = 1.0    # writes the output every dt_out ms
    Istim  = 60    # intensity of the stimulus
    tstim  = 1.0    # duration of the stimulus (in ms)
    tt     = TenTusscher(None, None, npt=100)
    U      = tt.initialize(dt=dt)
    plot_freq = int(dt_out/dt)  # writes the solution every plot_freq time steps
    URES      = []
    for jj in range(int(TEND/dt)):
        dU = tt.differentiate(U)
        if(jj<=int(tstim/dt)):
            U += dt*(dU+Istim)
        else:
            U += dt*dU
       # tt.states[0]=U
        if jj%plot_freq==0:
            URES.append(U)
import matplotlib.pyplot as plt
plt.plot(np.array(URES))
plt.show()
