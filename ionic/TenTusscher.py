from ten_tusscher_model_2006_IK1Ko_endo_units import sizeAlgebraic, createLegends, initConsts
import torch

class TenTusscher:
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
        rates = self.computes_rates(states=torch.cat((U, self.H), dim=1), constants=self.constants)

        dU = rates[0]
        dH = rates[1:]

        self.H += self.dt * dH
        return dU

    def computes_rates(self, states, constants):
        rates = torch.zeros_like(states)
        algebraic = torch.zeros((states.shape[0], sizeAlgebraic), device=self.device, dtype=self.dtype)





        return rates


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
