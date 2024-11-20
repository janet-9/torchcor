import torch

sizeAlgebraic = 3
sizeStates = 2
sizeConstants = 10

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (ms)"
    legend_algebraic[0] = "J_stim in component J_stim (per_ms)"
    legend_constants[0] = "IstimStart in component J_stim (ms)"
    legend_constants[1] = "IstimEnd in component J_stim (ms)"
    legend_constants[2] = "IstimAmplitude in component J_stim (per_ms)"
    legend_constants[3] = "IstimPeriod in component J_stim (ms)"
    legend_constants[4] = "IstimPulseDuration in component J_stim (ms)"
    legend_states[0] = "Vm in component membrane (dimensionless)"
    legend_algebraic[1] = "J_in in component J_in (per_ms)"
    legend_algebraic[2] = "J_out in component J_out (per_ms)"
    legend_constants[5] = "tau_in in component J_in (ms)"
    legend_states[1] = "h in component J_in_h_gate (dimensionless)"
    legend_constants[6] = "tau_open in component J_in_h_gate (ms)"
    legend_constants[7] = "tau_close in component J_in_h_gate (ms)"
    legend_constants[8] = "V_gate in component J_in_h_gate (dimensionless)"
    legend_constants[9] = "tau_out in component J_out (ms)"
    legend_rates[0] = "d/dt Vm in component membrane (dimensionless)"
    legend_rates[1] = "d/dt h in component J_in_h_gate (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 0
    constants[1] = 50000
    constants[2] = 0.2
    constants[3] = 500
    constants[4] = 1
    states[0] = 0.00000820413566106744
    constants[5] = 0.3
    states[1] = 0.8789655121804799
    constants[6] = 120.0
    constants[7] = 150.0
    constants[8] = 0.13
    constants[9] = 6.0
    return (states, constants)


class MitchellSchaeffer:
    def __init__(self, cell_model, device, dtype=torch.float64):
        self.cell_model = cell_model
        self.device = device
        self.dtype = dtype

        init_states, init_constants = initConsts()
        self.states = torch.tensor(init_states, device=device, dtype=dtype)
        self.constants = None

        self.name_constant_dict = OrderedDict()
        (_, _, _, legend_constants) = createLegends()
        for legend_constant, init_constant in zip(legend_constants, init_constants):
            constant_name = legend_constant.split()[0]
            self.name_constant_dict[constant_name] = init_constant

        self.H = None
        self.dt = None

    def initialize(self, n_nodes, dt):
        self.states = self.states.repeat(n_nodes, 1).clone()

        self.constants = torch.tensor(list(self.name_constant_dict.values()), device=self.device, dtype=self.dtype)
        U = self.states[:, 0].clone()
        self.H = self.states[:, 1:].clone()
        self.dt = dt

        return U

    def differentiate(self, U):
        self.states[:, 0] = U
        self.states[:, 1:] = self.H

        rates = self.compute_rates(states=self.states, constants=self.constants)

        dH = rates[:, 1:]
        self.H += self.dt * dH

        dU = rates[:, 0]

        return dU

    def compute_rates(self, states, constants):
        rates = torch.zeros_like(states)
        algebraic = torch.zeros((states.shape[0], self.cell_model.sizeAlgebraic), device=self.device, dtype=self.dtype)

        rates[:, 1] = torch.where(states[:, 0] < constants[8],
                                  (1.00000 - states[:, 1]) / constants[6],
                                  -states[:, 1] / constants[7])
        algebraic[:, 0] = 0.0
        algebraic[:, 1] = (states[:, 1] * ((torch.pow(states[:, 0], 2.00000)) * (1.00000 - states[:, 0]))) / constants[
            5]
        algebraic[:, 2] = -(states[:, 0] / constants[9])
        rates[:, 0] = algebraic[:, 1] + algebraic[:, 2] + algebraic[:, 0]

        return rates

if __name__ == "__main__":
    TEND   = 1000   # final time (in ms)
    dt     = 0.001  # time step
    dt_out = 1.0    # writes the output every dt_out ms
    Istim  = 100    # intensity of the stimulus
    tstim  = 1.0    # duration of the stimulus (in ms)
    tt     = TenTusscher(device=None, dtype=torch.float64)
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