# Size of variable arrays:
sizeAlgebraic = 3
sizeStates = 2
sizeConstants = 10
from math import *
from numpy import *

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

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    rates[1] = custom_piecewise([less(states[0] , constants[8]), (1.00000-states[1])/constants[6] , True, -states[1]/constants[7]])
    algebraic[0] = 0 # custom_piecewise([greater_equal(voi , constants[0]) & less_equal(voi , constants[1]) & less_equal((voi-constants[0])-floor((voi-constants[0])/constants[3])*constants[3] , constants[4]), constants[2] , True, 0.00000])
    algebraic[1] = (states[1]*((power(states[0], 2.00000))*(1.00000-states[0])))/constants[5]
    algebraic[2] = -(states[0]/constants[9])
    rates[0] = algebraic[1]+algebraic[2]+algebraic[0]
    return(rates)


if __name__ == "__main__":
    legend_states, legend_algebraics, _, legend_constants = createLegends()
    for legend_state in legend_states:
        state_name = legend_state.split()[0]
        print(state_name)

    tau_list = []
    inf_list = []
    alpha_list = []
    beta_list = []
    for legend_algebraic in legend_algebraics:
        algebraic_name = legend_algebraic.split()[0]
        if algebraic_name.startswith("tau"):
            tau_list.append(algebraic_name)
        if algebraic_name.endswith("inf"):
            inf_list.append(algebraic_name)
        if algebraic_name.startswith("alpha"):
            alpha_list.append(algebraic_name)
        if algebraic_name.startswith("beta"):
            beta_list.append(algebraic_name)
    print(len(tau_list), len(inf_list), len(alpha_list), len(beta_list))
