import torch
from ionic.cellml.courtemanche_ramirez_nattel_1998 import sizeAlgebraic, sizeStates, sizeConstants, initConsts, createLegends
from collections import OrderedDict


class CourtemancheRamirezNattel:
    def __init__(self, device, dtype=torch.float64):
        self.device = device
        self.dtype = dtype

        init_states, init_constants = initConsts()
        print(init_states)
        self.states = torch.tensor(init_states, device=device, dtype=dtype)
        self.constants = None

        (legend_states, legend_algebraics, _, legend_constants) = createLegends()

        self.name_constant_dict = OrderedDict()
        for legend_constant, init_constant in zip(legend_constants, init_constants):
            constant_name = legend_constant.split()[0]
            if "stim" in constant_name:
                self.name_constant_dict[constant_name] = 0
            else:
                self.name_constant_dict[constant_name] = init_constant

        gate_dict = {}
        for i, legend_state in enumerate(legend_states):
            # legend_state = legend_state.lower()
            # if "gate" in legend_state:
            state_name = legend_state.split()[0]
            gate_dict[state_name] = i
        self.gate_indices = list(gate_dict.values())
        self.non_gate_indices = [i for i in range(1, sizeStates) if i not in self.gate_indices]

        tau_dict = {}
        inf_dict = {}
        for i, legend_algebraic in enumerate(legend_algebraics):
            legend_algebraic = legend_algebraic.lower()
            algebraic_name = legend_algebraic.split()[0]
            if algebraic_name.startswith("tau"):
                tau_dict[algebraic_name.split("_")[1]] = i
            if algebraic_name.endswith("inf") or algebraic_name.endswith("infinity"):
                inf_dict[algebraic_name.split("_")[0]] = i

        # gating_variables = set(tau_dict.keys()).intersection(set(inf_dict.keys()))
        print(gate_dict, len(gate_dict))
        print(tau_dict, len(tau_dict))
        print(inf_dict, len(inf_dict))
        raise Exception()

        tau_dict = {key: tau_dict[key] for key in list(gate_dict.keys())}
        inf_dict = {key: inf_dict[key] for key in list(gate_dict.keys())}
        self.tau_indices = list(tau_dict.values())
        self.inf_indices = list(inf_dict.values())

        self.H = None
        self.dt = None

    def compute_rates(self, states, constants):
        rates = torch.zeros_like(states)
        algebraic = torch.zeros((states.shape[0], sizeAlgebraic), device=self.device, dtype=self.dtype)

        algebraic[:, 12] = torch.pow(1.00000+states[:, 12]/0.000350000, -1.00000)
        rates[:, 15] = (algebraic[:, 12]-states[:, 15])/constants[44]
        algebraic[:, 10] = torch.pow(1.00000+torch.exp((states[:, 0]+10.0000)/-8.00000), -1.00000)
        algebraic[:, 27] = torch.where(torch.abs(states[:, 0]+10.0000) < 1.00000e-10,
                                    4.57900/(1.00000+torch.exp((states[:, 0]+10.0000)/-6.24000)),
                                    (1.00000-torch.exp((states[:, 0]+10.0000)/-6.24000))/(0.0350000*(states[:, 0]+10.0000)*(1.00000+torch.exp((states[:, 0]+10.0000)/-6.24000))))
        rates[:, 13] = (algebraic[:, 10]-states[:, 13])/algebraic[:, 27]
        algebraic[:, 11] = torch.exp(-(states[:, 0]+28.0000)/6.90000)/(1.00000+torch.exp(-(states[:, 0]+28.0000)/6.90000))
        algebraic[:, 28] = 9.00000*(torch.pow(0.0197000*torch.exp(-(0.0337000 ** 2.00000)*(torch.pow(states[:, 0]+10.0000, 2.00000)))+0.0200000, -1.00000))
        rates[:, 14] = (algebraic[:, 11]-states[:, 14])/algebraic[:, 28]
        algebraic[:, 13] = torch.where(torch.abs(states[:, 0]-7.90000) < 1.00000e-10,
                                   (6.00000*0.200000)/1.30000,
                                   (6.00000*(1.00000-torch.exp(-(states[:, 0]-7.90000)/5.00000)))/((1.00000+0.300000*torch.exp(-(states[:, 0]-7.90000)/5.00000))*1.00000*(states[:, 0]-7.90000)))
        algebraic[:, 29] = 1.00000-torch.pow(1.00000+torch.exp(-(states[:, 0]-40.0000)/17.0000), -1.00000)
        rates[:, 19] = (algebraic[:, 29]-states[:, 19])/algebraic[:, 13]
        algebraic[:, 1] = torch.where(states[:, 0] < -47.1300,
                                   3.20000,
                                   (0.320000*(states[:, 0]+47.1300))/(1.00000-torch.exp(-0.100000*(states[:, 0]+47.1300))))
        algebraic[:, 18] = 0.0800000*torch.exp(-states[:, 0]/11.0000)
        algebraic[:, 31] = algebraic[:, 1]/(algebraic[:, 1]+algebraic[:, 18])
        algebraic[:, 41] = 1.00000/(algebraic[:, 1]+algebraic[:, 18])
        rates[:, 2] = (algebraic[:, 31]-states[:, 2])/algebraic[:, 41]
        algebraic[:, 2] = torch.where(states[:, 0] < -40.0000,
                                   0.135000*torch.exp((states[:, 0]+80.0000)/-6.80000),
                                   0.00000)
        algebraic[:, 19] = torch.where(states[:, 0] < -40.0000,
                                    3.56000*torch.exp(0.0790000*states[:, 0])+310000.*torch.exp(0.350000*states[:, 0]),
                                    1.00000/(0.130000*(1.00000+torch.exp((states[:, 0]+10.6600)/-11.1000))))
        algebraic[:, 32] = algebraic[:, 2]/(algebraic[:, 2]+algebraic[:, 19])
        algebraic[:, 42] = 1.00000/(algebraic[:, 2]+algebraic[:, 19])
        rates[:, 3] = (algebraic[:, 32]-states[:, 3])/algebraic[:, 42]
        algebraic[:, 3] = torch.where(states[:, 0] < -40.0000,
                                     ((-127140.*torch.exp(0.244400*states[:, 0])-3.47400e-05*torch.exp(-0.0439100*states[:, 0]))*(states[:, 0]+37.7800))/(1.00000+torch.exp(0.311000*(states[:, 0]+79.2300))),
                                     0.00000)
        algebraic[:, 20] = torch.where(states[:, 0] < -40.0000,
                                      (0.121200*torch.exp(-0.0105200*states[:, 0]))/(1.00000+torch.exp(-0.137800*(states[:, 0]+40.1400))),
                                      (0.300000*torch.exp(-2.53500e-07*states[:, 0]))/(1.00000+torch.exp(-0.100000*(states[:, 0]+32.0000))))
        algebraic[:, 33] = algebraic[:, 3]/(algebraic[:, 3]+algebraic[:, 20])
        algebraic[:, 43] = 1.00000/(algebraic[:, 3]+algebraic[:, 20])
        rates[:, 4] = (algebraic[:, 33]-states[:, 4])/algebraic[:, 43]
        algebraic[:, 4] = 0.650000*(torch.pow(torch.exp((states[:, 0]--10.0000)/-8.50000)+torch.exp(((states[:, 0]--10.0000)-40.0000)/-59.0000), -1.00000))
        algebraic[:, 21] = 0.650000*(torch.pow(2.50000+torch.exp(((states[:, 0]--10.0000)+72.0000)/17.0000), -1.00000))
        algebraic[:, 34] = (torch.pow(algebraic[:, 4]+algebraic[:, 21], -1.00000))/constants[13]
        algebraic[:, 44] = torch.pow(1.00000+torch.exp(((states[:, 0]--10.0000)+10.4700)/-17.5400), -1.00000)
        rates[:, 6] = (algebraic[:, 44]-states[:, 6])/algebraic[:, 34]
        algebraic[:, 5] = torch.pow(18.5300+1.00000*torch.exp(((states[:, 0]--10.0000)+103.700)/10.9500), -1.00000)
        algebraic[:, 22] = torch.pow(35.5600+1.00000*torch.exp(((states[:, 0]--10.0000)-8.74000)/-7.44000), -1.00000)
        algebraic[:, 35] = (torch.pow(algebraic[:, 5]+algebraic[:, 22], -1.00000))/constants[13]
        algebraic[:, 45] = torch.pow(1.00000+torch.exp(((states[:, 0]--10.0000)+33.1000)/5.30000), -1.00000)
        rates[:, 7] = (algebraic[:, 45]-states[:, 7])/algebraic[:, 35]
        algebraic[:, 6] = 0.650000*(torch.pow(torch.exp((states[:, 0]--10.0000)/-8.50000)+torch.exp(((states[:, 0]--10.0000)-40.0000)/-59.0000), -1.00000))
        algebraic[:, 23] = 0.650000*(torch.pow(2.50000+torch.exp(((states[:, 0]--10.0000)+72.0000)/17.0000), -1.00000))
        algebraic[:, 36] = (torch.pow(algebraic[:, 6]+algebraic[:, 23], -1.00000))/constants[13]
        algebraic[:, 46] = torch.pow(1.00000+torch.exp(((states[:, 0]--10.0000)+20.3000)/-9.60000), -1.00000)
        rates[:, 8] = (algebraic[:, 46]-states[:, 8])/algebraic[:, 36]
        algebraic[:, 7] = torch.pow(21.0000+1.00000*torch.exp(((states[:, 0]--10.0000)-195.000)/-28.0000), -1.00000)
        algebraic[:, 24] = 1.00000/torch.exp(((states[:, 0]--10.0000)-168.000)/-16.0000)
        algebraic[:, 37] = (torch.pow(algebraic[:, 7]+algebraic[:, 24], -1.00000))/constants[13]
        algebraic[:, 47] = torch.pow(1.00000+torch.exp(((states[:, 0]--10.0000)-109.450)/27.4800), -1.00000)
        rates[:, 9] = (algebraic[:, 47]-states[:, 9])/algebraic[:, 37]
        algebraic[:, 8] = torch.where(torch.abs(states[:, 0]+14.1000) < 1.00000e-10,
                                   0.00150000,
                                   (0.000300000*(states[:, 0]+14.1000))/(1.00000-torch.exp((states[:, 0]+14.1000)/-5.00000)))
        algebraic[:, 25] = torch.where(torch.abs(states[:, 0]-3.33280) < 1.00000e-10,
                                    0.000378361,
                                    (7.38980e-05*(states[:, 0]-3.33280))/(torch.exp((states[:, 0]-3.33280)/5.12370)-1.00000))
        algebraic[:, 38] = torch.pow(algebraic[:, 8]+algebraic[:, 25], -1.00000)
        algebraic[:, 48] = torch.pow(1.00000+torch.exp((states[:, 0]+14.1000)/-6.50000), -1.00000)
        rates[:, 10] = (algebraic[:, 48]-states[:, 10])/algebraic[:, 38]
        algebraic[:, 9] = torch.where(torch.abs(states[:, 0]-19.9000) < 1.00000e-10,
                                   0.000680000,
                                   (4.00000e-05*(states[:, 0]-19.9000))/(1.00000-torch.exp((states[:, 0]-19.9000)/-17.0000)))
        algebraic[:, 26] = torch.where(torch.abs(states[:, 0]-19.9000) < 1.00000e-10,
                                    0.000315000,
                                    (3.50000e-05*(states[:, 0]-19.9000))/(torch.exp((states[:, 0]-19.9000)/9.00000)-1.00000))
        algebraic[:, 39] = 0.500000*(torch.pow(algebraic[:, 9]+algebraic[:, 26], -1.00000))
        algebraic[:, 49] = torch.pow(1.00000+torch.exp((states[:, 0]-19.9000)/-12.7000), -0.500000)
        rates[:, 11] = (algebraic[:, 49]-states[:, 11])/algebraic[:, 39]
        algebraic[:, 40] = ((constants[0]*constants[1])/constants[2])*torch.log(constants[12]/states[:, 5])
        algebraic[:, 50] = (constants[3]*constants[11]*(states[:, 0]-algebraic[:, 40]))/(1.00000+torch.exp(0.0700000*(states[:, 0]+80.0000)))
        algebraic[:, 51] = constants[3]*constants[14]*(torch.pow(states[:, 6], 3.00000))*states[:, 7]*(states[:, 0]-algebraic[:, 40])
        algebraic[:, 52] = 0.00500000+0.0500000/(1.00000+torch.exp((states[:, 0]-15.0000)/-13.0000))
        algebraic[:, 53] = constants[3]*algebraic[:, 52]*(torch.pow(states[:, 8], 3.00000))*states[:, 9]*(states[:, 0]-algebraic[:, 40])
        algebraic[:, 54] = (constants[3]*constants[15]*states[:, 10]*(states[:, 0]-algebraic[:, 40]))/(1.00000+torch.exp((states[:, 0]+15.0000)/22.4000))
        algebraic[:, 55] = constants[3]*constants[16]*(torch.pow(states[:, 11], 2.00000))*(states[:, 0]-algebraic[:, 40])
        algebraic[:, 57] = torch.pow(1.00000+0.124500*torch.exp((-0.100000*constants[2]*states[:, 0])/(constants[0]*constants[1]))+0.0365000*constants[45]*torch.exp((-constants[2]*states[:, 0])/(constants[0]*constants[1])), -1.00000)
        algebraic[:, 58] = (((constants[3]*constants[20]*algebraic[:, 57]*1.00000)/(1.00000+torch.pow(constants[18]/states[:, 1], 1.50000)))*constants[12])/(constants[12]+constants[19])
        algebraic[:, 60] = constants[3]*constants[23]*(states[:, 0]-algebraic[:, 40])
        rates[:, 5] = (2.00000*algebraic[:, 58]-(algebraic[:, 50]+algebraic[:, 51]+algebraic[:, 53]+algebraic[:, 54]+algebraic[:, 55]+algebraic[:, 60]))/(constants[43]*constants[2])
        algebraic[:, 17] = ((constants[0]*constants[1])/constants[2])*torch.log(constants[10]/states[:, 1])
        algebraic[:, 30] = constants[3]*constants[9]*(torch.pow(states[:, 2], 3.00000))*states[:, 3]*states[:, 4]*(states[:, 0]-algebraic[:, 17])
        algebraic[:, 63] = (constants[3]*constants[25]*(torch.exp((constants[29]*constants[2]*states[:, 0])/(constants[0]*constants[1]))*(torch.pow(states[:, 1], 3.00000))*constants[24]-torch.exp(((constants[29]-1.00000)*constants[2]*states[:, 0])/(constants[0]*constants[1]))*(torch.pow(constants[10], 3.00000))*states[:, 12]))/((torch.pow(constants[26], 3.00000)+torch.pow(constants[10], 3.00000))*(constants[27]+constants[24])*(1.00000+constants[28]*torch.exp(((constants[29]-1.00000)*states[:, 0]*constants[2])/(constants[0]*constants[1]))))
        algebraic[:, 61] = constants[3]*constants[21]*(states[:, 0]-algebraic[:, 17])
        rates[:, 1] = (-3.00000*algebraic[:, 58]-(3.00000*algebraic[:, 63]+algebraic[:, 61]+algebraic[:, 30]))/(constants[43]*constants[2])
        algebraic[:, 0] = 0.0 # custom_piecewise([greater_equal(voi , constants[4]) & less_equal(voi , constants[5]) & less_equal((voi-constants[4])-floor((voi-constants[4])/constants[6])*constants[6] , constants[7]), constants[8] , True, 0.00000])
        algebraic[:, 56] = constants[3]*constants[17]*states[:, 13]*states[:, 14]*states[:, 15]*(states[:, 0]-65.0000)
        algebraic[:, 64] = (constants[3]*constants[30]*states[:, 12])/(0.000500000+states[:, 12])
        algebraic[:, 59] = ((constants[0]*constants[1])/(2.00000*constants[2]))*torch.log(constants[24]/states[:, 12])
        algebraic[:, 62] = constants[3]*constants[22]*(states[:, 0]-algebraic[:, 59])
        rates[:, 0] = -(algebraic[:, 30]+algebraic[:, 50]+algebraic[:, 51]+algebraic[:, 53]+algebraic[:, 54]+algebraic[:, 55]+algebraic[:, 61]+algebraic[:, 62]+algebraic[:, 58]+algebraic[:, 64]+algebraic[:, 63]+algebraic[:, 56]+algebraic[:, 0])/constants[3]
        algebraic[:, 65] = constants[31]*(torch.pow(states[:, 17], 2.00000))*states[:, 18]*states[:, 19]*(states[:, 16]-states[:, 12])
        algebraic[:, 67] = (states[:, 20]-states[:, 16])/constants[32]
        rates[:, 16] = (algebraic[:, 67]-algebraic[:, 65])*(torch.pow(1.00000+(constants[38]*constants[41])/(torch.pow(states[:, 16]+constants[41], 2.00000)), -1.00000))
        algebraic[:, 66] = 1000.00*(1.00000e-15*constants[47]*algebraic[:, 65]-(1.00000e-15/(2.00000*constants[2]))*(0.500000*algebraic[:, 56]-0.200000*algebraic[:, 63]))
        algebraic[:, 68] = torch.pow(1.00000+torch.exp(-(algebraic[:, 66]-3.41750e-13)/1.36700e-15), -1.00000)
        rates[:, 17] = (algebraic[:, 68]-states[:, 17])/constants[46]
        algebraic[:, 69] = 1.91000+2.09000*(torch.pow(1.00000+torch.exp(-(algebraic[:, 66]-3.41750e-13)/1.36700e-15), -1.00000))
        algebraic[:, 71] = 1.00000-torch.pow(1.00000+torch.exp(-(algebraic[:, 66]-6.83500e-14)/1.36700e-15), -1.00000)
        rates[:, 18] = (algebraic[:, 71]-states[:, 18])/algebraic[:, 69]
        algebraic[:, 70] = constants[33]/(1.00000+constants[34]/states[:, 12])
        algebraic[:, 72] = (constants[33]*states[:, 20])/constants[35]
        rates[:, 20] = algebraic[:, 70]-(algebraic[:, 72]+(algebraic[:, 67]*constants[47])/constants[48])
        algebraic[:, 73] = (2.00000*algebraic[:, 63]-(algebraic[:, 64]+algebraic[:, 56]+algebraic[:, 62]))/(2.00000*constants[43]*constants[2])+(constants[48]*(algebraic[:, 72]-algebraic[:, 70])+algebraic[:, 65]*constants[47])/constants[43]
        algebraic[:, 74] = 1.00000+(constants[37]*constants[40])/(torch.pow(states[:, 12]+constants[40], 2.00000))+(constants[36]*constants[39])/(torch.pow(states[:, 12]+constants[39], 2.00000))
        rates[:, 12] = algebraic[:, 73]/algebraic[:, 74]

        return rates

    def initialize(self, n_nodes, dt):
        self.states = self.states.repeat(n_nodes, 1).clone()

        self.constants = torch.tensor(list(self.name_constant_dict.values()), device=self.device, dtype=self.dtype)
        U = self.states[:, 0].clone()
        self.H = self.states[:, 1:].clone()
        self.dt = dt

        return U

    def differentiate(self, U):
        self.states[:, 0] = U

        rates, algebraic = self.compute_rates(states=self.states, constants=self.constants)

        # update states
        self.apply_rush_larsen(algebraic, self.dt)
        self.states[:, self.non_gate_indices] += self.dt * rates[:, self.non_gate_indices]

        dU = rates[:, 0]
        return dU

    def apply_rush_larsen(self, algebraic, dt):
        steady_states = algebraic[:, self.inf_indices]
        time_constants = algebraic[:, self.tau_indices]

        # Update gating variables using Rush-Larsen method
        self.states[:, self.gate_indices] = steady_states + (self.states[:, self.gate_indices] - steady_states) * torch.exp(-dt / time_constants)

    def default_constants(self):
        return dict(self.name_constant_dict)

    def reset_constant(self, name, value):
        self.name_constant_dict[name] = value

    def set_attribute(self, name, value):
        setattr(self, name, value)

    def get_attribute(self, name: str):
        return getattr(self, name, None)


if __name__ == "__main__":
    TEND   = 500   # final time (in ms)
    dt     = 0.001  # time step
    dt_out = 1.0    # writes the output every dt_out ms
    Istim  = 100    # intensity of the stimulus
    tstim  = 1.0    # duration of the stimulus (in ms)
    tt     = CourtemancheRamirezNattel(device=None, dtype=torch.float64)
    print(tt.name_constant_dict)
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