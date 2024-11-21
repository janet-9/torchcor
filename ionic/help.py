import torch

# Size of variable arrays:
sizeAlgebraic = 70
sizeStates = 19
sizeConstants = 53

def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_constants[0] = "R in component membrane (joule_per_mole_kelvin)"
    legend_constants[1] = "T in component membrane (kelvin)"
    legend_constants[2] = "F in component membrane (coulomb_per_millimole)"
    legend_constants[3] = "Cm in component membrane (microF)"
    legend_constants[4] = "V_c in component membrane (micrometre3)"
    legend_constants[5] = "stim_start in component membrane (millisecond)"
    legend_constants[6] = "stim_period in component membrane (millisecond)"
    legend_constants[7] = "stim_duration in component membrane (millisecond)"
    legend_constants[8] = "stim_amplitude in component membrane (picoA_per_picoF)"
    legend_constants[9] = "P_kna in component reversal_potentials (dimensionless)"
    legend_constants[10] = "K_o in component potassium_dynamics (millimolar)"
    legend_constants[11] = "Na_o in component sodium_dynamics (millimolar)"
    legend_constants[12] = "Ca_o in component calcium_dynamics (millimolar)"
    legend_constants[13] = "g_K1 in component inward_rectifier_potassium_current (nanoS_per_picoF)"
    legend_constants[14] = "g_Kr in component rapid_time_dependent_potassium_current (nanoS_per_picoF)"
    legend_constants[15] = "g_Ks in component slow_time_dependent_potassium_current (nanoS_per_picoF)"
    legend_constants[16] = "g_Na in component fast_sodium_current (nanoS_per_picoF)"
    legend_constants[17] = "g_bna in component sodium_background_current (nanoS_per_picoF)"
    legend_constants[18] = "g_CaL in component L_type_Ca_current (litre_per_farad_second)"
    legend_constants[19] = "g_bca in component calcium_background_current (nanoS_per_picoF)"
    legend_constants[20] = "g_to in component transient_outward_current (nanoS_per_picoF)"
    legend_constants[22] = "K_mk in component sodium_potassium_pump_current (millimolar)"
    legend_constants[23] = "K_mNa in component sodium_potassium_pump_current (millimolar)"
    legend_constants[24] = "K_NaCa in component sodium_calcium_exchanger_current (picoA_per_picoF)"
    legend_constants[25] = "K_sat in component sodium_calcium_exchanger_current (dimensionless)"
    legend_constants[26] = "alpha in component sodium_calcium_exchanger_current (dimensionless)"
    legend_constants[27] = "gamma in component sodium_calcium_exchanger_current (dimensionless)"
    legend_constants[28] = "Km_Ca in component sodium_calcium_exchanger_current (millimolar)"
    legend_constants[29] = "Km_Nai in component sodium_calcium_exchanger_current (millimolar)"
    legend_constants[30] = "g_pCa in component calcium_pump_current (picoA_per_picoF)"
    legend_constants[31] = "K_pCa in component calcium_pump_current (millimolar)"
    legend_constants[32] = "g_pK in component potassium_pump_current (nanoS_per_picoF)"
    legend_constants[34] = "k2_prime in component calcium_dynamics (per_millimolar_per_millisecond)"
    legend_constants[35] = "k3 in component calcium_dynamics (per_millisecond)"
    legend_constants[36] = "k4 in component calcium_dynamics (per_millisecond)"
    legend_constants[37] = "EC in component calcium_dynamics (millimolar)"
    legend_constants[38] = "max_sr in component calcium_dynamics (dimensionless)"
    legend_constants[39] = "min_sr in component calcium_dynamics (dimensionless)"
    legend_constants[40] = "V_rel in component calcium_dynamics (per_millisecond)"
    legend_constants[41] = "V_xfer in component calcium_dynamics (per_millisecond)"
    legend_constants[42] = "K_up in component calcium_dynamics (millimolar)"
    legend_constants[43] = "V_leak in component calcium_dynamics (per_millisecond)"
    legend_constants[44] = "Vmax_up in component calcium_dynamics (millimolar_per_millisecond)"
    legend_constants[45] = "Buf_c in component calcium_dynamics (millimolar)"
    legend_constants[46] = "K_buf_c in component calcium_dynamics (millimolar)"
    legend_constants[47] = "Buf_sr in component calcium_dynamics (millimolar)"
    legend_constants[48] = "K_buf_sr in component calcium_dynamics (millimolar)"
    legend_constants[49] = "Buf_ss in component calcium_dynamics (millimolar)"
    legend_constants[50] = "K_buf_ss in component calcium_dynamics (millimolar)"
    legend_constants[51] = "V_sr in component calcium_dynamics (micrometre3)"
    legend_constants[52] = "V_ss in component calcium_dynamics (micrometre3)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    states[0] = -86.709
    constants[0] = 8314.472
    constants[1] = 310
    constants[2] = 96485.3415
    constants[3] = 0.185
    constants[4] = 0.016404
    constants[5] = 10
    constants[6] = 1000
    constants[7] = 1
    constants[8] = 52
    constants[9] = 0.03
    constants[10] = 5.4
    constants[11] = 140
    states[1] = 138.4
    states[2] = 10.355
    constants[12] = 2
    states[3] = 0.00013
    constants[13] = 5.405
    constants[14] = 0.153
    states[4] = 0.00448
    states[5] = 0.476
    constants[15] = 0.392
    states[6] = 0.0087
    constants[16] = 14.838
    states[7] = 0.00155
    states[8] = 0.7573
    states[9] = 0.7225
    constants[17] = 0.00029
    constants[18] = 0.0000398
    states[10] = 0.00036
    states[11] = 3.164e-5
    states[12] = 0.8009
    states[13] = 0.9778
    states[14] = 0.9953
    constants[19] = 0.000592
    constants[20] = 0.073
    states[15] = 0.3212
    states[16] = 2.235e-8
    constants[21] = 2.724
    constants[22] = 1
    constants[23] = 40
    constants[24] = 1000
    constants[25] = 0.1
    constants[26] = 2.5
    constants[27] = 0.35
    constants[28] = 1.38
    constants[29] = 87.5
    constants[30] = 0.1238
    constants[31] = 0.0005
    constants[32] = 0.0146
    states[17] = 3.715
    states[18] = 0.9068
    constants[33] = 0.15
    constants[34] = 0.045
    constants[35] = 0.06
    constants[36] = 0.005
    constants[37] = 1.5
    constants[38] = 2.5
    constants[39] = 1
    constants[40] = 0.102
    constants[41] = 0.0038
    constants[42] = 0.00025
    constants[43] = 0.00036
    constants[44] = 0.006375
    constants[45] = 0.2
    constants[46] = 0.001
    constants[47] = 10
    constants[48] = 0.3
    constants[49] = 0.4
    constants[50] = 0.00025
    constants[51] = 0.001094
    constants[52] = 0.00005468
    return (states, constants)


class TenTusscher:
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

        algebraic[:, 7] = 1.00000/(1.00000+torch.exp((states[:, 0]+20.0000)/7.00000))
        algebraic[:, 20] = 1102.50*torch.exp(-(torch.pow(states[:, 0]+27.0000, 2.00000))/225.000)+200.000/(1.00000+torch.exp((13.0000-states[:, 0])/10.0000))+180.000/(1.00000+torch.exp((states[:, 0]+30.0000)/10.0000))+20.0000
        rates[:, 12] = (algebraic[:, 7]-states[:, 12])/algebraic[:, 20]
        algebraic[:, 8] = 0.670000/(1.00000+torch.exp((states[:, 0]+35.0000)/7.00000))+0.330000
        algebraic[:, 21] = 562.000*torch.exp(-(torch.pow(states[:, 0]+27.0000, 2.00000))/240.000)+31.0000/(1.00000+torch.exp((25.0000-states[:, 0])/10.0000))+80.0000/(1.00000+torch.exp((states[:, 0]+30.0000)/10.0000))
        rates[:, 13] = (algebraic[:, 8]-states[:, 13])/algebraic[:, 21]
        algebraic[:, 9] = 0.600000/(1.00000+torch.pow(states[:, 10]/0.0500000, 2.00000))+0.400000
        algebraic[:, 22] = 80.0000/(1.00000+torch.pow(states[:, 10]/0.0500000, 2.00000))+2.00000
        rates[:, 14] = (algebraic[:, 9]-states[:, 14])/algebraic[:, 22]
        algebraic[:, 10] = 1.00000/(1.00000+torch.exp((states[:, 0]+28.0000)/5.00000))
        algebraic[:, 23] = 1000.00*torch.exp(-(torch.pow(states[:, 0]+67.0000, 2.00000))/1000.00)+8.00000
        rates[:, 15] = (algebraic[:, 10]-states[:, 15])/algebraic[:, 23]
        algebraic[:, 11] = 1.00000/(1.00000+torch.exp((20.0000-states[:, 0])/6.00000))
        algebraic[:, 24] = 9.50000*torch.exp(-(torch.pow(states[:, 0]+40.0000, 2.00000))/1800.00)+0.800000
        rates[:, 16] = (algebraic[:, 11]-states[:, 16])/algebraic[:, 24]
        algebraic[:, 0] = 1.00000/(1.00000+torch.exp((-26.0000-states[:, 0])/7.00000))
        algebraic[:, 13] = 450.000/(1.00000+torch.exp((-45.0000-states[:, 0])/10.0000))
        algebraic[:, 26] = 6.00000/(1.00000+torch.exp((states[:, 0]+30.0000)/11.5000))
        algebraic[:, 34] = 1.00000*algebraic[:, 13]*algebraic[:, 26]
        rates[:, 4] = (algebraic[:, 0]-states[:, 4])/algebraic[:, 34]
        algebraic[:, 1] = 1.00000/(1.00000+torch.exp((states[:, 0]+88.0000)/24.0000))
        algebraic[:, 14] = 3.00000/(1.00000+torch.exp((-60.0000-states[:, 0])/20.0000))
        algebraic[:, 27] = 1.12000/(1.00000+torch.exp((states[:, 0]-60.0000)/20.0000))
        algebraic[:, 35] = 1.00000*algebraic[:, 14]*algebraic[:, 27]
        rates[:, 5] = (algebraic[:, 1]-states[:, 5])/algebraic[:, 35]
        algebraic[:, 2] = 1.00000/(1.00000+torch.exp((-5.00000-states[:, 0])/14.0000))
        algebraic[:, 15] = 1400.00/(torch.pow(1.00000+torch.exp((5.00000-states[:, 0])/6.00000), 1.0/2))
        algebraic[:, 28] = 1.00000/(1.00000+torch.exp((states[:, 0]-35.0000)/15.0000))
        algebraic[:, 36] = 1.00000*algebraic[:, 15]*algebraic[:, 28]+80.0000
        rates[:, 6] = (algebraic[:, 2]-states[:, 6])/algebraic[:, 36]
        algebraic[:, 3] = 1.00000/(torch.pow(1.00000+torch.exp((-56.8600-states[:, 0])/9.03000), 2.00000))
        algebraic[:, 16] = 1.00000/(1.00000+torch.exp((-60.0000-states[:, 0])/5.00000))
        algebraic[:, 29] = 0.100000/(1.00000+torch.exp((states[:, 0]+35.0000)/5.00000))+0.100000/(1.00000+torch.exp((states[:, 0]-50.0000)/200.000))
        algebraic[:, 37] = 1.00000*algebraic[:, 16]*algebraic[:, 29]
        rates[:, 7] = (algebraic[:, 3]-states[:, 7])/algebraic[:, 37]
        algebraic[:, 4] = 1.00000/(torch.pow(1.00000+torch.exp((states[:, 0]+71.5500)/7.43000), 2.00000))
        algebraic[:, 17] = torch.where(states[:, 0] < -40.0000,
                                    0.0570000*torch.exp(-(states[:, 0]+80.0000)/6.80000),
                                    0.00000)
        algebraic[:, 30] = torch.where(states[:, 0] < -40.0000,
                                    2.70000*torch.exp(0.0790000*states[:, 0])+310000.*torch.exp(0.348500*states[:, 0]),
                                    0.770000/(0.130000*(1.00000+torch.exp((states[:, 0]+10.6600)/-11.1000))))
        algebraic[:, 38] = 1.00000/(algebraic[:, 17]+algebraic[:, 30])
        rates[:, 8] = (algebraic[:, 4]-states[:, 8])/algebraic[:, 38]
        algebraic[:, 5] = 1.00000/(torch.pow(1.00000+torch.exp((states[:, 0]+71.5500)/7.43000), 2.00000))
        algebraic[:, 18] = torch.where(states[:, 0] < -40.0000,
                                    (((-25428.0*torch.exp(0.244400*states[:, 0])-6.94800e-06*torch.exp(-0.0439100*states[:, 0]))*(states[:, 0]+37.7800))/1.00000)/(1.00000+torch.exp(0.311000*(states[:, 0]+79.2300))),
                                    0.00000)
        algebraic[:, 31] = torch.where(states[:, 0] < -40.0000,
                                    (0.0242400*torch.exp(-0.0105200*states[:, 0]))/(1.00000+torch.exp(-0.137800*(states[:, 0]+40.1400))),
                                    (0.600000*torch.exp(0.0570000*states[:, 0]))/(1.00000+torch.exp(-0.100000*(states[:, 0]+32.0000))))
        algebraic[:, 39] = 1.00000/(algebraic[:, 18]+algebraic[:, 31])
        rates[:, 9] = (algebraic[:, 5]-states[:, 9])/algebraic[:, 39]
        algebraic[:, 6] = 1.00000/(1.00000+torch.exp((-8.00000-states[:, 0])/7.50000))
        algebraic[:, 19] = 1.40000/(1.00000+torch.exp((-35.0000-states[:, 0])/13.0000))+0.250000
        algebraic[:, 32] = 1.40000/(1.00000+torch.exp((states[:, 0]+5.00000)/5.00000))
        algebraic[:, 40] = 1.00000/(1.00000+torch.exp((50.0000-states[:, 0])/20.0000))
        algebraic[:, 42] = 1.00000*algebraic[:, 19]*algebraic[:, 32]+algebraic[:, 40]
        rates[:, 11] = (algebraic[:, 6]-states[:, 11])/algebraic[:, 42]
        algebraic[:, 55] = ((((constants[21]*constants[10])/(constants[10]+constants[22]))*states[:, 2])/(states[:, 2]+constants[23]))/(1.00000+0.124500*torch.exp((-0.100000*states[:, 0]*constants[2])/(constants[0]*constants[1]))+0.0353000*torch.exp((-states[:, 0]*constants[2])/(constants[0]*constants[1])))
        algebraic[:, 25] = ((constants[0]*constants[1])/constants[2])*torch.log(constants[11]/states[:, 2])
        algebraic[:, 50] = constants[16]*(torch.pow(states[:, 7], 3.00000))*states[:, 8]*states[:, 9]*(states[:, 0]-algebraic[:, 25])
        algebraic[:, 51] = constants[17]*(states[:, 0]-algebraic[:, 25])
        algebraic[:, 56] = (constants[24]*(torch.exp((constants[27]*states[:, 0]*constants[2])/(constants[0]*constants[1]))*(torch.pow(states[:, 2], 3.00000))*constants[12]-torch.exp(((constants[27]-1.00000)*states[:, 0]*constants[2])/(constants[0]*constants[1]))*(torch.pow(constants[11], 3.00000))*states[:, 3]*constants[26]))/((torch.pow(constants[29], 3.00000)+torch.pow(constants[11], 3.00000))*(constants[28]+constants[12])*(1.00000+constants[25]*torch.exp(((constants[27]-1.00000)*states[:, 0]*constants[2])/(constants[0]*constants[1]))))
        rates[:, 2] = ((-1.00000*(algebraic[:, 50]+algebraic[:, 51]+3.00000*algebraic[:, 55]+3.00000*algebraic[:, 56]))/(1.00000*constants[4]*constants[2]))*constants[3]
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
        rates[:, 0] = (-1.00000/1.00000)*(algebraic[:, 47]+algebraic[:, 54]+algebraic[:, 48]+algebraic[:, 49]+algebraic[:, 52]+algebraic[:, 55]+algebraic[:, 50]+algebraic[:, 51]+algebraic[:, 56]+algebraic[:, 53]+algebraic[:, 58]+algebraic[:, 57]+algebraic[:, 12])
        rates[:, 1] = ((-1.00000*((algebraic[:, 47]+algebraic[:, 54]+algebraic[:, 48]+algebraic[:, 49]+algebraic[:, 58]+algebraic[:, 12])-2.00000*algebraic[:, 55]))/(1.00000*constants[4]*constants[2]))*constants[3]
        algebraic[:, 59] = constants[44]/(1.00000+(torch.pow(constants[42], 2.00000))/(torch.pow(states[:, 3], 2.00000)))
        algebraic[:, 60] = constants[43]*(states[:, 17]-states[:, 3])
        algebraic[:, 61] = constants[41]*(states[:, 10]-states[:, 3])
        algebraic[:, 63] = 1.00000/(1.00000+(constants[45]*constants[46])/(torch.pow(states[:, 3]+constants[46], 2.00000)))
        rates[:, 3] = algebraic[:, 63]*((((algebraic[:, 60]-algebraic[:, 59])*constants[51])/constants[4]+algebraic[:, 61])-(1.00000*((algebraic[:, 53]+algebraic[:, 57])-2.00000*algebraic[:, 56])*constants[3])/(2.00000*1.00000*constants[4]*constants[2]))
        algebraic[:, 62] = constants[38]-(constants[38]-constants[39])/(1.00000+torch.pow(constants[37]/states[:, 17], 2.00000))
        algebraic[:, 65] = constants[34]*algebraic[:, 62]
        rates[:, 18] = -algebraic[:, 65]*states[:, 10]*states[:, 18]+constants[36]*(1.00000-states[:, 18])
        algebraic[:, 64] = constants[33]/algebraic[:, 62]
        algebraic[:, 66] = (algebraic[:, 64]*(torch.pow(states[:, 10], 2.00000))*states[:, 18])/(constants[35]+algebraic[:, 64]*(torch.pow(states[:, 10], 2.00000)))
        algebraic[:, 67] = constants[40]*algebraic[:, 66]*(states[:, 17]-states[:, 10])
        algebraic[:, 68] = 1.00000/(1.00000+(constants[47]*constants[48])/(torch.pow(states[:, 17]+constants[48], 2.00000)))
        rates[:, 17] = algebraic[:, 68]*(algebraic[:, 59]-(algebraic[:, 67]+algebraic[:, 60]))
        algebraic[:, 69] = 1.00000/(1.00000+(constants[49]*constants[50])/(torch.pow(states[:, 10]+constants[50], 2.00000)))
        rates[:, 10] = algebraic[:, 69]*(((-1.00000*algebraic[:, 52]*constants[3])/(2.00000*1.00000*constants[52]*constants[2])+(algebraic[:, 67]*constants[51])/constants[52])-(algebraic[:, 61]*constants[4])/constants[52])

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


def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_algebraic[47] = "i_K1 in component inward_rectifier_potassium_current (picoA_per_picoF)"
    legend_algebraic[54] = "i_to in component transient_outward_current (picoA_per_picoF)"
    legend_algebraic[48] = "i_Kr in component rapid_time_dependent_potassium_current (picoA_per_picoF)"
    legend_algebraic[49] = "i_Ks in component slow_time_dependent_potassium_current (picoA_per_picoF)"
    legend_algebraic[52] = "i_CaL in component L_type_Ca_current (picoA_per_picoF)"
    legend_algebraic[55] = "i_NaK in component sodium_potassium_pump_current (picoA_per_picoF)"
    legend_algebraic[50] = "i_Na in component fast_sodium_current (picoA_per_picoF)"
    legend_algebraic[51] = "i_b_Na in component sodium_background_current (picoA_per_picoF)"
    legend_algebraic[56] = "i_NaCa in component sodium_calcium_exchanger_current (picoA_per_picoF)"
    legend_algebraic[53] = "i_b_Ca in component calcium_background_current (picoA_per_picoF)"
    legend_algebraic[58] = "i_p_K in component potassium_pump_current (picoA_per_picoF)"
    legend_algebraic[57] = "i_p_Ca in component calcium_pump_current (picoA_per_picoF)"
    legend_algebraic[12] = "i_Stim in component membrane (picoA_per_picoF)"
    legend_algebraic[25] = "E_Na in component reversal_potentials (millivolt)"
    legend_algebraic[33] = "E_K in component reversal_potentials (millivolt)"
    legend_algebraic[41] = "E_Ks in component reversal_potentials (millivolt)"
    legend_algebraic[43] = "E_Ca in component reversal_potentials (millivolt)"
    legend_algebraic[46] = "xK1_inf in component inward_rectifier_potassium_current (dimensionless)"
    legend_algebraic[44] = "alpha_K1 in component inward_rectifier_potassium_current (dimensionless)"
    legend_algebraic[45] = "beta_K1 in component inward_rectifier_potassium_current (dimensionless)"
    legend_algebraic[0] = "xr1_inf in component rapid_time_dependent_potassium_current_Xr1_gate (dimensionless)"
    legend_algebraic[13] = "alpha_xr1 in component rapid_time_dependent_potassium_current_Xr1_gate (dimensionless)"
    legend_algebraic[26] = "beta_xr1 in component rapid_time_dependent_potassium_current_Xr1_gate (dimensionless)"
    legend_algebraic[34] = "tau_xr1 in component rapid_time_dependent_potassium_current_Xr1_gate (millisecond)"
    legend_algebraic[1] = "xr2_inf in component rapid_time_dependent_potassium_current_Xr2_gate (dimensionless)"
    legend_algebraic[14] = "alpha_xr2 in component rapid_time_dependent_potassium_current_Xr2_gate (dimensionless)"
    legend_algebraic[27] = "beta_xr2 in component rapid_time_dependent_potassium_current_Xr2_gate (dimensionless)"
    legend_algebraic[35] = "tau_xr2 in component rapid_time_dependent_potassium_current_Xr2_gate (millisecond)"
    legend_algebraic[2] = "xs_inf in component slow_time_dependent_potassium_current_Xs_gate (dimensionless)"
    legend_algebraic[15] = "alpha_xs in component slow_time_dependent_potassium_current_Xs_gate (dimensionless)"
    legend_algebraic[28] = "beta_xs in component slow_time_dependent_potassium_current_Xs_gate (dimensionless)"
    legend_algebraic[36] = "tau_xs in component slow_time_dependent_potassium_current_Xs_gate (millisecond)"
    legend_algebraic[16] = "alpha_m in component fast_sodium_current_m_gate (dimensionless)"
    legend_algebraic[29] = "beta_m in component fast_sodium_current_m_gate (dimensionless)"
    legend_algebraic[37] = "tau_m in component fast_sodium_current_m_gate (millisecond)"
    legend_algebraic[4] = "h_inf in component fast_sodium_current_h_gate (dimensionless)"
    legend_algebraic[17] = "alpha_h in component fast_sodium_current_h_gate (per_millisecond)"
    legend_algebraic[30] = "beta_h in component fast_sodium_current_h_gate (per_millisecond)"
    legend_algebraic[38] = "tau_h in component fast_sodium_current_h_gate (millisecond)"
    legend_algebraic[5] = "j_inf in component fast_sodium_current_j_gate (dimensionless)"
    legend_algebraic[18] = "alpha_j in component fast_sodium_current_j_gate (per_millisecond)"
    legend_algebraic[31] = "beta_j in component fast_sodium_current_j_gate (per_millisecond)"
    legend_algebraic[39] = "tau_j in component fast_sodium_current_j_gate (millisecond)"
    legend_algebraic[19] = "alpha_d in component L_type_Ca_current_d_gate (dimensionless)"
    legend_algebraic[32] = "beta_d in component L_type_Ca_current_d_gate (dimensionless)"
    legend_algebraic[40] = "gamma_d in component L_type_Ca_current_d_gate (millisecond)"
    legend_algebraic[42] = "tau_d in component L_type_Ca_current_d_gate (millisecond)"
    legend_algebraic[7] = "f_inf in component L_type_Ca_current_f_gate (dimensionless)"
    legend_algebraic[20] = "tau_f in component L_type_Ca_current_f_gate (millisecond)"
    legend_algebraic[8] = "f2_inf in component L_type_Ca_current_f2_gate (dimensionless)"
    legend_algebraic[21] = "tau_f2 in component L_type_Ca_current_f2_gate (millisecond)"
    legend_algebraic[9] = "fCass_inf in component L_type_Ca_current_fCass_gate (dimensionless)"
    legend_algebraic[22] = "tau_fCass in component L_type_Ca_current_fCass_gate (millisecond)"
    legend_algebraic[10] = "s_inf in component transient_outward_current_s_gate (dimensionless)"
    legend_algebraic[23] = "tau_s in component transient_outward_current_s_gate (millisecond)"
    legend_algebraic[11] = "r_inf in component transient_outward_current_r_gate (dimensionless)"
    legend_algebraic[24] = "tau_r in component transient_outward_current_r_gate (millisecond)"
    legend_algebraic[67] = "i_rel in component calcium_dynamics (millimolar_per_millisecond)"
    legend_algebraic[59] = "i_up in component calcium_dynamics (millimolar_per_millisecond)"
    legend_algebraic[60] = "i_leak in component calcium_dynamics (millimolar_per_millisecond)"
    legend_algebraic[61] = "i_xfer in component calcium_dynamics (millimolar_per_millisecond)"
    legend_algebraic[66] = "O in component calcium_dynamics (dimensionless)"
    legend_algebraic[64] = "k1 in component calcium_dynamics (per_millimolar2_per_millisecond)"
    legend_algebraic[65] = "k2 in component calcium_dynamics (per_millimolar_per_millisecond)"
    legend_algebraic[62] = "kcasr in component calcium_dynamics (dimensionless)"
    legend_algebraic[63] = "Ca_i_bufc in component calcium_dynamics (dimensionless)"
    legend_algebraic[68] = "Ca_sr_bufsr in component calcium_dynamics (dimensionless)"
    legend_algebraic[69] = "Ca_ss_bufss in component calcium_dynamics (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)