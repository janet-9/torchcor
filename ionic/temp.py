from cellml.ten_tusscher_model_2006_IK1Ko_endo_units import sizeAlgebraic, createLegends, initConsts, computeRates
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
        # self.states[1:] = self.H
        # print(self.states.tolist())
        rates = computeRates(None, states=self.states.tolist(), constants=self.constants)

        dU = rates[0]
        dH = rates[1:]


        self.H += self.dt * np.array(dH)
        # print(dU)
        return dU



if __name__ == "__main__":
   URES=[]
   tt = TenTusscher(None, None, npt=100)
   dt=0.01
   U = tt.initialize(dt=dt)
   Istim=100

   for jj in range(int(600000)):
       if(jj>100):
           Istim=0
       dU = tt.differentiate(U)
       U += dt*(dU+Istim)
       # print(dU)
       # print(U)
       # tt.states[0]=U
       if jj%100==0:
        URES.append(U)
import matplotlib.pyplot as plt
plt.plot(np.array(URES))
plt.show()
