import torch
from ionic.cellml.ten_tusscher_model_2006_IK1Ko_endo_units import sizeStates, initConsts, createLegends
from collections import OrderedDict


def construct_LUT(device):
    V = torch.arange(-800, 800, 0.05).to(device)

    a7 = 1.00000/(1.00000+torch.exp((V + 20.0000)/7.00000))
    a20 = 1102.50*torch.exp(-(torch.pow(V+27.0000, 2.00000))/225.000)+200.000/(1.00000+torch.exp((13.0000-V)/10.0000))+180.000/(1.00000+torch.exp((V+30.0000)/10.0000))+20.0000
    a8 = 0.670000/(1.00000+torch.exp((V+35.0000)/7.00000))+0.330000
    a21 = 562.000*torch.exp(-(torch.pow(V+27.0000, 2.00000))/240.000)+31.0000/(1.00000+torch.exp((25.0000-V)/10.0000))+80.0000/(1.00000+torch.exp((V+30.0000)/10.0000))
    a10 = 1.00000/(1.00000+torch.exp((V+28.0000)/5.00000))
    a23 = 1000.00*torch.exp(-(torch.pow(V+67.0000, 2.00000))/1000.00)+8.00000
    a11 = 1.00000/(1.00000+torch.exp((20.0000-V)/6.00000))
    a24 = 9.50000*torch.exp(-(torch.pow(V+40.0000, 2.00000))/1800.00)+0.800000
    a0 = 1.00000/(1.00000+torch.exp((-26.0000-V)/7.00000))
    a13 = 450.000/(1.00000+torch.exp((-45.0000-V)/10.0000))
    a26 = 6.00000/(1.00000+torch.exp((V+30.0000)/11.5000))
    a34 = 1.00000*a13*a26
    a1 = 1.00000/(1.00000+torch.exp((V+88.0000)/24.0000))
    a14 = 3.00000/(1.00000+torch.exp((-60.0000-V)/20.0000))
    a27 = 1.12000/(1.00000+torch.exp((V-60.0000)/20.0000))
    a35 = 1.00000*a14*a27
    a2 = 1.00000/(1.00000+torch.exp((-5.00000-V)/14.0000))
    a15 = 1400.00/(torch.pow(1.00000+torch.exp((5.00000-V)/6.00000), 1.0/2))
    a28 = 1.00000/(1.00000+torch.exp((V-35.0000)/15.0000))
    a36 = 1.00000*a15*a28+80.0000
    a3 = 1.00000/(torch.pow(1.00000+torch.exp((-56.8600-V)/9.03000), 2.00000))
    a16 = 1.00000/(1.00000+torch.exp((-60.0000-V)/5.00000))
    a29 = 0.100000/(1.00000+torch.exp((V+35.0000)/5.00000))+0.100000/(1.00000+torch.exp((V-50.0000)/200.000))
    a37 = 1.00000*a16*a29
    a4 = 1.00000/(torch.pow(1.00000+torch.exp((V+71.5500)/7.43000), 2.00000))
    a17 = torch.where(V < -40.0000,
                      0.0570000*torch.exp(-(V+80.0000)/6.80000),
                      0.00000)
    a30 = torch.where(V < -40.0000,
                     2.70000*torch.exp(0.0790000*V)+310000.*torch.exp(0.348500*V),
                     0.770000/(0.130000*(1.00000+torch.exp((V+10.6600)/-11.1000))))
    a38 = 1.00000/(a17+a30)
    a5 = 1.00000/(torch.pow(1.00000+torch.exp((V+71.5500)/7.43000), 2.00000))
    a18 = torch.where(V < -40.0000,
                     (((-25428.0*torch.exp(0.244400*V)-6.94800e-06*torch.exp(-0.0439100*V))*(V+37.7800))/1.00000)/(1.00000+torch.exp(0.311000*(V+79.2300))),
                     0.00000)
    a31 = torch.where(V < -40.0000,
                     (0.0242400*torch.exp(-0.0105200*V))/(1.00000+torch.exp(-0.137800*(V+40.1400))),
                     (0.600000*torch.exp(0.0570000*V))/(1.00000+torch.exp(-0.100000*(V+32.0000))))
    a39 = 1.00000/(a18+a31)
    a6 = 1.00000/(1.00000+torch.exp((-8.00000-V)/7.50000))
    a19 = 1.40000/(1.00000+torch.exp((-35.0000-V)/13.0000))+0.250000
    a32 = 1.40000/(1.00000+torch.exp((V+5.00000)/5.00000))
    a40 = 1.00000/(1.00000+torch.exp((50.0000-V)/20.0000))

    # [32000, 36]
    V_LUT = torch.stack([a7, a20, a8, a21, a10, a23, a11, a24, a0, a13, a26, a34, a1, a14, a27, a35, a2, a15,
                         a28, a36, a3, a16, a29, a37, a4, a17, a30, a38, a5, a18, a31, a39, a6, a19, a32, a40], dim=1)

    return V_LUT

@torch.jit.script
def compute_algebraic(states, constants, V_LUT):
    algebraic = torch.zeros((states.shape[0], 70), device=states.device, dtype=states.dtype)
    V = states[:, 0]
    indices = (V + 800) / 0.05
    lower_indices = torch.floor(indices).long()
    upper_indices = lower_indices + 1

    V1 = -800 + lower_indices * 0.05
    V2 = -800 + upper_indices * 0.05

    weights = (V - V1) / (V2 - V1)
    V_ROW = (1 - weights).unsqueeze(1) * V_LUT[lower_indices] + weights.unsqueeze(1) * V_LUT[upper_indices] # [637480, 36]

    algebraic[:, 7] = V_ROW[:, 0]
    algebraic[:, 20] = V_ROW[:, 1]
    algebraic[:, 8] = V_ROW[:, 2]
    algebraic[:, 21] = V_ROW[:, 3]
    algebraic[:, 9] = 0.600000/(1.00000+torch.pow(states[:, 10]/0.0500000, 2.00000))+0.400000
    algebraic[:, 22] = 80.0000/(1.00000+torch.pow(states[:, 10]/0.0500000, 2.00000))+2.00000
    algebraic[:, 10] = V_ROW[:, 4]
    algebraic[:, 23] = V_ROW[:, 5]
    algebraic[:, 11] = V_ROW[:, 6]
    algebraic[:, 24] = V_ROW[:, 7]
    algebraic[:, 0] = V_ROW[:, 8]
    algebraic[:, 13] = V_ROW[:, 9]
    algebraic[:, 26] = V_ROW[:, 10]
    algebraic[:, 34] = V_ROW[:, 11]
    algebraic[:, 1] = V_ROW[:, 12]
    algebraic[:, 14] = V_ROW[:, 13]
    algebraic[:, 27] = V_ROW[:, 14]
    algebraic[:, 35] = V_ROW[:, 15]
    algebraic[:, 2] = V_ROW[:, 16]
    algebraic[:, 15] = V_ROW[:, 17]
    algebraic[:, 28] = V_ROW[:, 18]
    algebraic[:, 36] = V_ROW[:, 19]
    algebraic[:, 3] = V_ROW[:, 20]
    algebraic[:, 16] = V_ROW[:, 21]
    algebraic[:, 29] = V_ROW[:, 22]
    algebraic[:, 37] = V_ROW[:, 23]
    algebraic[:, 4] = V_ROW[:, 24]
    algebraic[:, 17] = V_ROW[:, 25]
    algebraic[:, 30] = V_ROW[:, 26]
    algebraic[:, 38] = V_ROW[:, 27]
    algebraic[:, 5] = V_ROW[:, 28]
    algebraic[:, 18] = V_ROW[:, 29]
    algebraic[:, 31] = V_ROW[:, 30]
    algebraic[:, 39] = V_ROW[:, 31]
    algebraic[:, 6] = V_ROW[:, 32]
    algebraic[:, 19] = V_ROW[:, 33]
    algebraic[:, 32] = V_ROW[:, 34]
    algebraic[:, 40] = V_ROW[:, 35]
    algebraic[:, 42] = 1.00000*algebraic[:, 19]*algebraic[:, 32]+algebraic[:, 40]
    algebraic[:, 55] = ((((constants[21]*constants[10])/(constants[10]+constants[22]))*states[:, 2])/(states[:, 2]+constants[23]))/(1.00000+0.124500*torch.exp((-0.100000*states[:, 0]*constants[2])/(constants[0]*constants[1]))+0.0353000*torch.exp((-states[:, 0]*constants[2])/(constants[0]*constants[1])))
    algebraic[:, 25] = ((constants[0]*constants[1])/constants[2])*torch.log(constants[11]/states[:, 2])
    algebraic[:, 50] = constants[16]*(torch.pow(states[:, 7], 3.00000))*states[:, 8]*states[:, 9]*(states[:, 0]-algebraic[:, 25])
    algebraic[:, 51] = constants[17]*(V-algebraic[:, 25])
    algebraic[:, 56] = (constants[24]*(torch.exp((constants[27]*states[:, 0]*constants[2])/(constants[0]*constants[1]))*(torch.pow(states[:, 2], 3.00000))*constants[12]-torch.exp(((constants[27]-1.00000)*states[:, 0]*constants[2])/(constants[0]*constants[1]))*(torch.pow(constants[11], 3.00000))*states[:, 3]*constants[26]))/((torch.pow(constants[29], 3.00000)+torch.pow(constants[11], 3.00000))*(constants[28]+constants[12])*(1.00000+constants[25]*torch.exp(((constants[27]-1.00000)*states[:, 0]*constants[2])/(constants[0]*constants[1]))))
    algebraic[:, 33] = ((constants[0]*constants[1])/constants[2])*torch.log(constants[10]/states[:, 1])
    algebraic[:, 44] = 0.100000/(1.00000+torch.exp(0.0600000*((states[:, 0]-algebraic[:, 33])-200.000)))
    algebraic[:, 45] = (3.00000*torch.exp(0.000200000*((states[:, 0]-algebraic[:, 33])+100.000))+torch.exp(0.100000*((states[:, 0]-algebraic[:, 33])-10.0000)))/(1.00000+torch.exp(-0.500000*(states[:, 0]-algebraic[:, 33])))
    algebraic[:, 46] = algebraic[:, 44]/(algebraic[:, 44]+algebraic[:, 45])
    algebraic[:, 47] = constants[13]*algebraic[:, 46]*(torch.pow(constants[10]/5.40000, 1.0/2))*(states[:, 0]-algebraic[:, 33])
    algebraic[:, 54] = constants[20]*states[:, 16]*states[:, 15]*(states[:, 0]-algebraic[:, 33])
    algebraic[:, 48] = constants[14]*(torch.pow(constants[10]/5.40000, 1.0/2))*states[:, 4]*states[:, 5]*(states[:, 0]-algebraic[:, 33])
    algebraic[:, 41] = ((constants[0]*constants[1])/constants[2])*torch.log((constants[10]+constants[9]*constants[11])/(states[:, 1]+constants[9]*states[:, 2]))
    algebraic[:, 49] = constants[15]*(torch.pow(states[:, 6], 2.00000))*(states[:, 0]-algebraic[:, 41])
    algebraic[:, 52] = (((constants[18]*states[:, 11]*states[:, 12]*states[:, 13]*states[:, 14]*4.00000*(states[:, 0]-15.0000)*(torch.pow(constants[2], 2.00000)))/(constants[0]*constants[1]))*(0.250000*states[:, 10]*torch.exp((2.00000*(states[:, 0]-15.0000)*constants[2])/(constants[0]*constants[1]))-constants[12]))/(torch.exp((2.00000*(states[:, 0]-15.0000)*constants[2])/(constants[0]*constants[1]))-1.00000)
    algebraic[:, 43] = ((0.500000*constants[0]*constants[1])/constants[2])*torch.log(constants[12]/states[:, 3])
    algebraic[:, 53] = constants[19]*(states[:, 0]-algebraic[:, 43])
    algebraic[:, 58] = (constants[32]*(states[:, 0]-algebraic[:, 33]))/(1.00000+torch.exp((25.0000-states[:, 0])/5.98000))
    algebraic[:, 57] = (constants[30]*states[:, 3])/(states[:, 3]+constants[31])
    algebraic[:, 12] = 0.0 #custom_piecewise([greater_equal(voi-floor(voi/constants[6])*constants[6] , constants[5]) & less_equal(voi-floor(voi/constants[6])*constants[6] , constants[5]+constants[7]), -constants[8] , True, 0.00000])
    algebraic[:, 59] = constants[44]/(1.00000+(torch.pow(constants[42], 2.00000))/(torch.pow(states[:, 3], 2.00000)))
    algebraic[:, 60] = constants[43]*(states[:, 17]-states[:, 3])
    algebraic[:, 61] = constants[41]*(states[:, 10]-states[:, 3])
    algebraic[:, 63] = 1.00000/(1.00000+(constants[45]*constants[46])/(torch.pow(states[:, 3]+constants[46], 2.00000)))
    algebraic[:, 62] = constants[38]-(constants[38]-constants[39])/(1.00000+torch.pow(constants[37]/states[:, 17], 2.00000))
    algebraic[:, 65] = constants[34]*algebraic[:, 62]
    algebraic[:, 64] = constants[33]/algebraic[:, 62]
    algebraic[:, 66] = (algebraic[:, 64]*(torch.pow(states[:, 10], 2.00000))*states[:, 18])/(constants[35]+algebraic[:, 64]*(torch.pow(states[:, 10], 2.00000)))
    algebraic[:, 67] = constants[40]*algebraic[:, 66]*(states[:, 17]-states[:, 10])
    algebraic[:, 68] = 1.00000/(1.00000+(constants[47]*constants[48])/(torch.pow(states[:, 17]+constants[48], 2.00000)))
    algebraic[:, 69] = 1.00000/(1.00000+(constants[49]*constants[50])/(torch.pow(states[:, 10]+constants[50], 2.00000)))

    return algebraic

@torch.jit.script
def compute_rates(algebraic, states, constants):
    rates = torch.zeros_like(states)

    rates[:, 12] = (algebraic[:, 7]-states[:, 12])/algebraic[:, 20]
    rates[:, 13] = (algebraic[:, 8]-states[:, 13])/algebraic[:, 21]
    rates[:, 14] = (algebraic[:, 9]-states[:, 14])/algebraic[:, 22]
    rates[:, 15] = (algebraic[:, 10]-states[:, 15])/algebraic[:, 23]
    rates[:, 16] = (algebraic[:, 11]-states[:, 16])/algebraic[:, 24]
    rates[:, 4] = (algebraic[:, 0]-states[:, 4])/algebraic[:, 34]
    rates[:, 5] = (algebraic[:, 1]-states[:, 5])/algebraic[:, 35]
    rates[:, 6] = (algebraic[:, 2]-states[:, 6])/algebraic[:, 36]
    rates[:, 7] = (algebraic[:, 3]-states[:, 7])/algebraic[:, 37]
    rates[:, 8] = (algebraic[:, 4]-states[:, 8])/algebraic[:, 38]
    rates[:, 9] = (algebraic[:, 5]-states[:, 9])/algebraic[:, 39]
    rates[:, 11] = (algebraic[:, 6]-states[:, 11])/algebraic[:, 42]
    rates[:, 2] = ((-1.00000*(algebraic[:, 50]+algebraic[:, 51]+3.00000*algebraic[:, 55]+3.00000*algebraic[:, 56]))/(1.00000*constants[4]*constants[2]))*constants[3]
    rates[:, 0] = (-1.00000/1.00000)*(algebraic[:, 47]+algebraic[:, 54]+algebraic[:, 48]+algebraic[:, 49]+algebraic[:, 52]+algebraic[:, 55]+algebraic[:, 50]+algebraic[:, 51]+algebraic[:, 56]+algebraic[:, 53]+algebraic[:, 58]+algebraic[:, 57]+algebraic[:, 12])
    rates[:, 1] = ((-1.00000*((algebraic[:, 47]+algebraic[:, 54]+algebraic[:, 48]+algebraic[:, 49]+algebraic[:, 58]+algebraic[:, 12])-2.00000*algebraic[:, 55]))/(1.00000*constants[4]*constants[2]))*constants[3]
    rates[:, 3] = algebraic[:, 63]*((((algebraic[:, 60]-algebraic[:, 59])*constants[51])/constants[4]+algebraic[:, 61])-(1.00000*((algebraic[:, 53]+algebraic[:, 57])-2.00000*algebraic[:, 56])*constants[3])/(2.00000*1.00000*constants[4]*constants[2]))
    rates[:, 18] = -algebraic[:, 65]*states[:, 10]*states[:, 18]+constants[36]*(1.00000-states[:, 18])
    rates[:, 17] = algebraic[:, 68]*(algebraic[:, 59]-(algebraic[:, 67]+algebraic[:, 60]))
    rates[:, 10] = algebraic[:, 69]*(((-1.00000*algebraic[:, 52]*constants[3])/(2.00000*1.00000*constants[52]*constants[2])+(algebraic[:, 67]*constants[51])/constants[52])-(algebraic[:, 61]*constants[4])/constants[52])

    return rates

class TenTusscherPanfilov:
    def __init__(self, device, dtype=torch.float64):
        self.device = device
        self.dtype = dtype

        init_states, init_constants = initConsts()

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
            legend_state = legend_state.lower()
            if "gate" in legend_state:
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

        tau_dict = {key: tau_dict[key] for key in list(gate_dict.keys())}
        inf_dict = {key: inf_dict[key] for key in list(gate_dict.keys())}
        self.tau_indices = list(tau_dict.values())
        self.inf_indices = list(inf_dict.values())

        self.H = None
        self.dt = None

        self.LUT = construct_LUT(device)

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

    def compute_rates(self, states, constants):
        algebraic = compute_algebraic(states, constants, self.V_LUT)
        rates = compute_rates(algebraic, states, constants)

        return rates, algebraic

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
    TEND   = 1000   # final time (in ms)
    dt     = 0.001  # time step
    dt_out = 1.0    # writes the output every dt_out ms
    Istim  = 100    # intensity of the stimulus
    tstim  = 1.0    # duration of the stimulus (in ms)
    tt     = TenTusscherPanfilov(device=None, dtype=torch.float64)
    print(tt.default_constants())
    # print(tt.)
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