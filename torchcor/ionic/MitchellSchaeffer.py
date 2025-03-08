import torch
from torchcor.ionic.cellml import mitchell_schaeffer_2003
from torchcor.ionic.base import BaseCellModel, BaseCellModelRL


class MitchellSchaeffer(BaseCellModel):
    def __init__(self, device, dtype=torch.float32):
        super().__init__(mitchell_schaeffer_2003, device, dtype)
        self.name = "MitchellSchaeffer"

    def compute_rates(self, states, constants):
        rates = torch.zeros_like(states)
        algebraic = torch.zeros((states.shape[0], self.cell_model.sizeAlgebraic), device=self.device, dtype=self.dtype)

        rates[:, 1] = torch.where(states[:, 0] < constants[8],
                                  (1.00000-states[:, 1])/constants[6],
                                  -states[:, 1]/constants[7])
        algebraic[:, 0] = 0.0
        algebraic[:, 1] = (states[:, 1]*((torch.pow(states[:, 0], 2.00000))*(1.00000-states[:, 0])))/constants[5]
        algebraic[:, 2] = -(states[:, 0]/constants[9])
        rates[:, 0] = algebraic[:, 1]+algebraic[:, 2]+algebraic[:, 0]

        return rates


class MitchellSchaefferRL(BaseCellModel):
    def __init__(self, device, dtype=torch.float64):
        super().__init__(mitchell_schaeffer_2003, device, dtype)

    def differentiate(self, U):
        # Update states for membrane potential and gating variable
        self.states[:, 0] = U  # Vm
        h = self.states[:, 1]  # h

        # Compute rates and algebraic terms
        rates = self.compute_rates(states=self.states, constants=self.constants)

        # Rush-Larsen update for h
        tau_open = self.constants[6]
        tau_close = self.constants[7]
        V_gate = self.constants[8]

        # Determine steady-state h_inf and tau_h
        h_inf = torch.where(U < V_gate, 1.0, 0.0)
        tau_h = torch.where(U < V_gate, tau_open, tau_close)

        # Update h using Rush-Larsen method
        self.states[:, 1] = h_inf + (h - h_inf) * torch.exp(-self.dt / tau_h)

        # Update Vm (membrane potential)
        dU = rates[:, 0]
        return dU

    def compute_rates(self, states, constants):
        rates = torch.zeros_like(states)
        algebraic = torch.zeros((states.shape[0], self.cell_model.sizeAlgebraic), device=self.device, dtype=self.dtype)

        rates[:, 1] = torch.where(states[:, 0] < constants[8],
                                  (1.00000-states[:, 1])/constants[6],
                                  -states[:, 1]/constants[7])
        algebraic[:, 0] = 0.0
        algebraic[:, 1] = (states[:, 1]*((torch.pow(states[:, 0], 2.00000))*(1.00000-states[:, 0])))/constants[5]
        algebraic[:, 2] = -(states[:, 0]/constants[9])
        rates[:, 0] = algebraic[:, 1]+algebraic[:, 2]+algebraic[:, 0]

        return rates


if __name__ == "__main__":
    TEND   = 500   # final time (in ms)
    dt     = 0.001  # time step
    dt_out = 1.0    # writes the output every dt_out ms
    Istim  = 100    # intensity of the stimulus
    tstim  = 1.0    # duration of the stimulus (in ms)
    tt     = MitchellSchaeffer(device=None, dtype=torch.float64)
    print(tt.default_constants())
    # U      = tt.initialize(n_nodes=1, dt=dt)
    # plot_freq = int(dt_out/dt)  # writes the solution every plot_freq time steps
    # URES      = []
    # for jj in range(int(TEND/dt)):
    #     dU = tt.differentiate(U)
    #     if(jj<=int(tstim/dt)):
    #         U += dt*(dU+Istim)
    #     else:
    #         U += dt*dU
    #    # tt.states[0]=U
    #     if jj%plot_freq==0:
    #         URES.append(U.item())
    # import matplotlib.pyplot as plt
    # plt.plot(URES)
    # plt.show()