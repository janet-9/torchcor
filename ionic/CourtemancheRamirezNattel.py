import torch
from dataclasses import dataclass
from math import log, exp, expm1

C_B1a = 3.79138232501097e-05
C_B1b = 0.0811764705882353
C_B1c = 0.00705882352941176
C_B1d = 0.00537112496043221
C_B1e = 11.5
C_Fn1 = 9.648e-13
C_Fn2 = 2.5910306809e-13
C_dCa_rel = 8.
C_dCaup = 0.0869565217391304
Ca_rel_init = 1.49
Ca_up_init = 1.49
Cai_init = 1.02e-1
F = 96.4867
K_Q10 = 3.
K_up = 0.00092
Ki_init = 139.0
KmCa = 1.38
KmCmdn = 0.00238
KmCsqn = 0.8
KmKo = 1.5
KmNa = 87.5
KmNa3 = 669921.875
KmNai = 10.
KmTrpn = 0.0005
Nai = 11.2
R = 8.3143
T = 310.
V_init = -81.2
Volcell = 20100.
Voli = 13668.
Volrel = 96.48
Volup = 1109.52
d_init = 1.37e-4
f_Ca_init = 0.775
f_init = 0.999
gamma = 0.35
h_init = 0.965
j_init = 0.978
k_rel = 30.
k_sat = 0.1
m_init = 2.91e-3
maxCmdn = 0.05
maxCsqn = 10.
maxTrpn = 0.07
oa_init = 3.04e-2
oi_init = 0.999
tau_f_Ca = 2.
tau_tr = 180.
tau_u = 8.
u_init = 0.
ua_init = 4.96e-3
ui_init = 0.999
v_init = 1.
w_init = 0.999
xr_init = 3.29e-5
xs_init = 1.87e-2


@dataclass
class Parameters:
    ACh = 0.000001
    Cao = 1.8
    Cm = 100.
    GACh = 0.
    GCaL = 0.1238
    GK1 = 0.09
    GKr = 0.0294
    GKs = 0.129
    GNa = 7.8
    GbCa = 0.00113
    GbNa = 0.000674
    Gto = 0.1652
    Ko = 5.4
    Nao = 140.
    factorGKur = 1.
    factorGrel = 1.
    factorGtr = 1.
    factorGup = 1.
    factorhGate = 0.
    factormGate = 0.
    factoroaGate = 0.
    factorxrGate = 1.
    maxCaup = 15.
    maxINaCa = 1600.
    maxINaK = 0.60
    maxIpCa = 0.275
    maxIup = 0.005

@dataclass(frozen=True)
class StatesIndex:
    Ca_rel = 0
    Ca_up = 1
    Cai = 2
    Ki = 3
    d = 4
    f = 5
    f_Ca = 6
    h = 7
    j = 8
    m = 9
    oa = 10
    oi = 11
    u = 12
    ua = 13
    ui = 14
    v = 15
    w = 16
    xr = 17
    xs = 18
    n_total = 19

@dataclass(frozen=True)
class Cai_TableIndex:
    carow_1_idx = 0
    carow_2_idx = 1
    carow_3_idx = 2
    conCa_idx = 3
    f_Ca_rush_larsen_A_idx = 4
    NROWS = 5


@dataclass(frozen=True)
class V_TableIndex:
  GKur_idx = 0
  INaK_idx = 1
  IbNa_idx = 2
  d_rush_larsen_A_idx = 3
  d_rush_larsen_B_idx = 4
  f_rush_larsen_A_idx = 5
  f_rush_larsen_B_idx = 6
  h_rush_larsen_A_idx = 7
  h_rush_larsen_B_idx = 8
  j_rush_larsen_A_idx = 9
  j_rush_larsen_B_idx = 10
  m_rush_larsen_A_idx = 11
  m_rush_larsen_B_idx = 12
  oa_rush_larsen_A_idx = 13
  oa_rush_larsen_B_idx = 14
  oi_rush_larsen_A_idx = 15
  oi_rush_larsen_B_idx = 16
  ua_rush_larsen_A_idx = 17
  ua_rush_larsen_B_idx = 18
  ui_rush_larsen_A_idx = 19
  ui_rush_larsen_B_idx = 20
  vrow_29_idx = 21
  vrow_31_idx = 22
  vrow_32_idx = 23
  vrow_36_idx = 24
  vrow_7_idx = 25
  w_rush_larsen_A_idx = 26
  w_rush_larsen_B_idx = 27
  xr_rush_larsen_A_idx = 28
  xr_rush_larsen_B_idx = 29
  xs_rush_larsen_A_idx = 30
  xs_rush_larsen_B_idx = 31
  NROWS = 32


@dataclass(frozen=True)
class fn_TableIndex:
  u_rush_larsen_A_idx = 0
  v_rush_larsen_A_idx = 1
  v_rush_larsen_B_idx = 2
  NROWS = 3

@dataclass(frozen=True)
class Cai_TableParam:
    mn = 3e-4
    mx = 30.0
    res = 3e-4
    step = 1 / res
    mn_idx = 0
    mx_idx = int((mx - mn) * step) - 1

@dataclass(frozen=True)
class V_TableParam:
    mn = -200
    mx = 200
    res = 0.1
    step = 1 / res
    mn_idx = 0
    mx_idx = int((mx - mn) * step) - 1

@dataclass(frozen=True)
class fn_TableParam:
    mn = -2e-11
    mx = 10.0e-11
    res = 2e-15
    step = 1 / res
    mn_idx = 0
    mx_idx = int((mx - mn) * step) - 1


def interpolate(X: torch.Tensor, table, tp):
    X = torch.clamp(X, tp.mn, tp.mx)
    idx = ((X - tp.mn) * tp.step).to(torch.long)
    lower_idx = torch.clamp(idx, 0, tp.mx_idx - 1)
    higher_idx = lower_idx + 1
    lower_pos = lower_idx * tp.res + tp.mn
    w = ((X - lower_pos) / tp.res).unsqueeze(1)
    return (1 - w) * table[lower_idx] + w * table[higher_idx]



class CourtemancheRamirezNattel:
    def __init__(self, dt, device, dtype=torch.float64):
        self.si = StatesIndex()
        self.sv = None

        self.rush_larsen_A = None
        self.rush_larsen_B = None

        self.Cai_tab = None
        self.V_tab = None
        self.fn_tab = None

        self.Cai_ti = Cai_TableIndex()
        self.V_ti = V_TableIndex()
        self.fn_ti = fn_TableIndex()

        self.dt = dt
        self.device = device
        self.dtype = dtype


    def construct_tables(self):
        p = Parameters()

        # Define the constants that depend on the parameters.
        E_Na = ((R*T)/F)*(log((p.Nao/Nai)))
        f_Ca_rush_larsen_C = expm1(((-self.dt)/tau_f_Ca))
        sigma = ((exp((p.Nao/67.3)))-1.)/7.
        u_rush_larsen_C = expm1(((-self.dt)/tau_u))

        # construct Cai Lookup Table
        Cai_tp = Cai_TableParam()
        Cai = torch.arange(Cai_tp.mn, Cai_tp.mx, Cai_tp.res).to(self.device).to(self.dtype)
        Cai_tab = torch.zeros((Cai.shape[0], self.Cai_ti.NROWS)).to(self.device).to(self.dtype)

        Cai_tab[:, self.Cai_ti.conCa_idx] = Cai / 1000
        Cai_tab[:, self.Cai_ti.carow_1_idx] = (p.factorGup*p.maxIup)/(1.+(K_up/Cai_tab[:, self.Cai_ti.conCa_idx]))
        Cai_tab[:, self.Cai_ti.carow_2_idx] = (((((((maxTrpn*KmTrpn)/(Cai_tab[:, self.Cai_ti.conCa_idx]+KmTrpn))/(Cai_tab[:, self.Cai_ti.conCa_idx]+KmTrpn))+(((maxCmdn*KmCmdn)/(Cai_tab[:, self.Cai_ti.conCa_idx]+KmCmdn))/(Cai_tab[:, self.Cai_ti.conCa_idx]+KmCmdn)))+1.)/C_B1c)/1000.)
        Cai_tab[:, self.Cai_ti.carow_3_idx] = (((p.maxIpCa*Cai_tab[:, self.Cai_ti.conCa_idx])/(0.0005+Cai_tab[:, self.Cai_ti.conCa_idx]))-(((((p.GbCa*R)*T)/2.)/F)*(torch.log((p.Cao/Cai_tab[:, self.Cai_ti.conCa_idx])))))
        f_Ca_inf = 1./(1.+(Cai_tab[:, self.Cai_ti.conCa_idx]/0.00035))
        Cai_tab[:, self.Cai_ti.f_Ca_rush_larsen_A_idx] = (-f_Ca_inf)*f_Ca_rush_larsen_C

        self.Cai_tab = Cai_tab

        # construct Cai Lookup Table
        V_tp = V_TableParam()
        V = torch.arange(V_tp.mn, V_tp.mx, V_tp.res).to(self.device).to(self.dtype)
        V_tab = torch.zeros((V.shape[0], self.V_ti.NROWS)).to(self.device).to(self.dtype)

        V_tab[:, self.V_ti.GKur_idx] = (0.005+(0.05/(1.+(torch.exp(((15.-(V))/13.))))))
        V_tab[:, self.V_ti.IbNa_idx] = (p.GbNa * (V - E_Na))
        a_h = torch.where(V>=-40., 0.0, (0.135 * (torch.exp((((V+80.) - p.factorhGate) / -6.8)))))
        a_j = torch.where(V<-40., ((((-127140.*(torch.exp((0.2444*V))))-((3.474e-5*(torch.exp((-0.04391*V))))))*(V+37.78))/(1.+(torch.exp((0.311*(V+79.23)))))), 0.)
        a_m = torch.where(V==-47.13, 3.2, ((0.32*(V+47.13))/(1.-((torch.exp((-0.1*(V+47.13))))))))
        aa_oa = (0.65 / ((torch.exp(((V+10.)/-8.5))) + (torch.exp(((30. - V) / 59.0)))))
        aa_oi = (1./(18.53+(torch.exp(((V+113.7)/10.95)))))
        aa_ua = (0.65/((torch.exp(((V+10.)/-8.5)))+(torch.exp(((V-(30.))/-59.0)))))
        aa_ui = (1. / (21. + (torch.exp(((V - 185.) / -28.)))))
        aa_xr = (p.factorxrGate*((0.0003*(V+14.1))/(1.-((torch.exp(((V+14.1)/-5.)))))))
        aa_xs = ((4.e-5*(V-(19.9)))/(1.-((torch.exp(((19.9-(V))/17.))))))
        b_h = torch.where((V>=-40.), ((1./0.13)/(1.+(torch.exp(((-(V+10.66))/11.1))))), ((3.56*(torch.exp((0.079*V))))+(3.1e5*(torch.exp((0.35*V))))))
        b_j = torch.where((V>=-40.), ((0.3*(torch.exp((-2.535e-7*V))))/(1.+(torch.exp((-0.1*(V+32.)))))), ((0.1212*(torch.exp((-0.01052*V))))/(1.+(torch.exp((-0.1378*(V+40.14)))))))
        b_m = (0.08 * (torch.exp(((-(V - p.factormGate)) / 11.))))
        bb_oa = (0.65/(2.5+(torch.exp(((V+82.)/17.)))))
        bb_oi = (1./(35.56+(torch.exp(((-(V+1.26))/7.44)))))
        bb_ua = (0.65/(2.5+(torch.exp(((V+82.)/17.)))))
        bb_ui = (torch.exp(((V - 158.) / 16.)))
        bb_xr = ((1.0/p.factorxrGate) * ((7.3898e-5 * (V - 3.3328)) / ((torch.exp(((V - 3.3328) / 5.1237))) - (1.))))
        bb_xs = ((3.5e-5 * (V - 19.9)) / ((torch.exp(((V - 19.9) / 9.))) - 1.))
        d_inf = (1./(1.+(torch.exp(((-(V+10.))/8.)))))
        f_NaK = (1./((1.+(0.1245*(torch.exp(((((-0.1*F)*V)/R)/T)))))+((0.0365*sigma)*(torch.exp(((((-F)*V)/R)/T))))))
        f_inf = (1./(1.+(torch.exp(((V+28.)/6.9)))))
        oa_inf = (1. / (1. + (torch.exp(((-((V+20.47) - p.factoroaGate)) / 17.54)))))
        oi_inf = (1./(1.+(torch.exp(((V+43.1)/5.3)))))
        tau_d = torch.where((V==-10.), (((1./6.24)/0.035)/2.), ((((1.-((torch.exp(((-(V+10.))/6.24)))))/0.035)/(V+10.))/(1.+(torch.exp(((-(V+10.))/6.24))))))
        tau_f = (9./((0.0197*(torch.exp((((-0.0337*0.0337)*(V+10.))*(V+10.)))))+0.02))
        tau_w = (((6.*(1.-((torch.exp(((7.9-(V))/5.))))))/(1.+(0.3*(torch.exp(((7.9-(V))/5.))))))/(V-(7.9)))
        ua_inf = (1./(1.+(torch.exp(((V+30.3)/-9.6)))))
        ui_inf = (1./(1.+(torch.exp(((V-(99.45))/27.48)))))
        V_tab[:, self.V_ti.vrow_29_idx] = (p.GCaL*(V-(65.)))
        V_tab[:, self.V_ti.vrow_31_idx] = ((((((((p.maxINaCa*(torch.exp(((((gamma*F)*V)/R)/T))))*Nai)*Nai)*Nai)*p.Cao)/(KmNa3+((p.Nao*p.Nao)*p.Nao)))/(KmCa+p.Cao))/(1.+(k_sat*(torch.exp((((((gamma-(1.))*F)*V)/R)/T))))))
        V_tab[:, self.V_ti.vrow_32_idx] = ((((((((p.maxINaCa*(torch.exp((((((gamma-(1.))*F)*V)/R)/T))))*p.Nao)*p.Nao)*p.Nao)/(KmNa3+((p.Nao*p.Nao)*p.Nao)))/(KmCa+p.Cao))/(1.+(k_sat*(torch.exp((((((gamma-(1.))*F)*V)/R)/T))))))/1000.)
        V_tab[:, self.V_ti.vrow_36_idx] = (V*p.GbCa)
        V_tab[:, self.V_ti.vrow_7_idx] = (p.GNa * (V - E_Na))
        w_inf = (1. - (1. / (1. + (torch.exp(((40. - V) / 17.))))))
        xr_inf = (1./(1.+(torch.exp(((V+14.1)/-6.5)))))
        xs_inf = (1. / (torch.sqrt((1. + (torch.exp(((V - 19.9) / -12.7)))))))
        V_tab[:, self.V_ti.INaK_idx] = ((((p.maxINaK*f_NaK)/(1.+(pow((KmNai/Nai),1.5))))*p.Ko)/(p.Ko+KmKo))
        V_tab[:, self.V_ti.d_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_d)))
        d_rush_larsen_C = (torch.expm1(((-self.dt)/tau_d)))
        V_tab[:, self.V_ti.f_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_f)))
        f_rush_larsen_C = (torch.expm1(((-self.dt)/tau_f)))
        V_tab[:, self.V_ti.h_rush_larsen_A_idx] = (((-a_h)/(a_h+b_h))*(torch.expm1(((-self.dt)*(a_h+b_h)))))
        V_tab[:, self.V_ti.h_rush_larsen_B_idx] = (torch.exp(((-self.dt)*(a_h+b_h))))
        V_tab[:, self.V_ti.j_rush_larsen_A_idx] = (((-a_j)/(a_j+b_j))*(torch.expm1(((-self.dt)*(a_j+b_j)))))
        V_tab[:, self.V_ti.j_rush_larsen_B_idx] = (torch.exp(((-self.dt)*(a_j+b_j))))
        V_tab[:, self.V_ti.m_rush_larsen_A_idx] = (((-a_m)/(a_m+b_m))*(torch.expm1(((-self.dt)*(a_m+b_m)))))
        V_tab[:, self.V_ti.m_rush_larsen_B_idx] = (torch.exp(((-self.dt)*(a_m+b_m))))
        tau_oa = ((1./(aa_oa+bb_oa))/K_Q10)
        tau_oi = ((1./(aa_oi+bb_oi))/K_Q10)
        tau_ua = ((1./(aa_ua+bb_ua))/K_Q10)
        tau_ui = ((1./(aa_ui+bb_ui))/K_Q10)
        tau_xr = (1./(aa_xr+bb_xr))
        tau_xs = (0.5/(aa_xs+bb_xs))
        V_tab[:, self.V_ti.w_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_w)))
        w_rush_larsen_C = (torch.expm1(((-self.dt)/tau_w)))
        V_tab[:, self.V_ti.d_rush_larsen_A_idx] = ((-d_inf)*d_rush_larsen_C)
        V_tab[:, self.V_ti.f_rush_larsen_A_idx] = ((-f_inf)*f_rush_larsen_C)
        V_tab[:, self.V_ti.oa_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_oa)))
        oa_rush_larsen_C = (torch.expm1(((-self.dt)/tau_oa)))
        V_tab[:, self.V_ti.oi_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_oi)))
        oi_rush_larsen_C = (torch.expm1(((-self.dt)/tau_oi)))
        V_tab[:, self.V_ti.ua_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_ua)))
        ua_rush_larsen_C = (torch.expm1(((-self.dt)/tau_ua)))
        V_tab[:, self.V_ti.ui_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_ui)))
        ui_rush_larsen_C = (torch.expm1(((-self.dt)/tau_ui)))
        V_tab[:, self.V_ti.w_rush_larsen_A_idx] = ((-w_inf)*w_rush_larsen_C)
        V_tab[:, self.V_ti.xr_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_xr)))
        xr_rush_larsen_C = (torch.expm1(((-self.dt)/tau_xr)))
        V_tab[:, self.V_ti.xs_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_xs)))
        xs_rush_larsen_C = (torch.expm1(((-self.dt)/tau_xs)))
        V_tab[:, self.V_ti.oa_rush_larsen_A_idx] = ((-oa_inf)*oa_rush_larsen_C)
        V_tab[:, self.V_ti.oi_rush_larsen_A_idx] = ((-oi_inf)*oi_rush_larsen_C)
        V_tab[:, self.V_ti.ua_rush_larsen_A_idx] = ((-ua_inf)*ua_rush_larsen_C)
        V_tab[:, self.V_ti.ui_rush_larsen_A_idx] = ((-ui_inf)*ui_rush_larsen_C)
        V_tab[:, self.V_ti.xr_rush_larsen_A_idx] = ((-xr_inf)*xr_rush_larsen_C)
        V_tab[:, self.V_ti.xs_rush_larsen_A_idx] = ((-xs_inf)*xs_rush_larsen_C)

        self.V_tab = V_tab

        # Create the fn lookup table
        fn_tp = fn_TableParam()
        fn = torch.arange(fn_tp.mn, fn_tp.mx, fn_tp.res).to(self.device).to(self.dtype)
        fn_tab = torch.zeros((fn.shape[0], self.fn_ti.NROWS)).to(self.device).to(self.dtype)

        tau_v = (1.91+(2.09/(1.+(torch.exp(((3.4175e-13-fn)/13.67e-16))))))
        u_inf = (1./(1.+(torch.exp(((3.4175e-13-fn)/13.67e-16)))))
        v_inf = (1. - (1. / (1. + (torch.exp(((6.835e-14 - fn) / 13.67e-16))))))
        fn_tab[:, self.fn_ti.u_rush_larsen_A_idx] = ((-u_inf)*u_rush_larsen_C)
        fn_tab[:, self.fn_ti.v_rush_larsen_B_idx] = (torch.exp(((-self.dt)/tau_v)))
        v_rush_larsen_C = (torch.expm1(((-self.dt)/tau_v)))
        fn_tab[:, self.fn_ti.v_rush_larsen_A_idx] = ((-v_inf)*v_rush_larsen_C)

        self.fn_tab = fn_tab

    def initialize(self, n_nodes):
        V = torch.full((n_nodes,), V_init, device=self.device, dtype=self.dtype)

        self.sv = torch.zeros((n_nodes, self.si.n_total), device=self.device, dtype=self.dtype)
        self.sv[:, self.si.Ca_rel] = Ca_rel_init
        self.sv[:, self.si.Ca_up] = Ca_up_init
        self.sv[:, self.si.Cai] = Cai_init
        self.sv[:, self.si.Ki] = Ki_init
        self.sv[:, self.si.d] = d_init
        self.sv[:, self.si.f] = f_init
        self.sv[:, self.si.f_Ca] = f_Ca_init
        self.sv[:, self.si.h] = h_init
        self.sv[:, self.si.j] = j_init
        self.sv[:, self.si.m] = m_init
        self.sv[:, self.si.oa] = oa_init
        self.sv[:, self.si.oi] = oi_init
        self.sv[:, self.si.u] = u_init
        self.sv[:, self.si.ua] = ua_init
        self.sv[:, self.si.ui] = ui_init
        self.sv[:, self.si.v] = v_init
        self.sv[:, self.si.w] = w_init
        self.sv[:, self.si.xr] = xr_init
        self.sv[:, self.si.xs] = xs_init

        self.rush_larsen_A = torch.zeros((n_nodes, 15), device=self.device, dtype=self.dtype)
        self.rush_larsen_B = torch.zeros((n_nodes, 15), device=self.device, dtype=self.dtype)
        self.rush_larsen_B[: 2] = exp(((-self.dt)/tau_f_Ca))   # f_Ca
        self.rush_larsen_B[: 8] = exp(((-self.dt)/tau_u))      # u

        return V


    def differentiate(self, V):
        p = Parameters()

        V_row = interpolate(V, self.V_tab, V_TableParam())
        Cai_row = interpolate(self.sv[:, self.si.Cai], self.Cai_tab, Cai_TableParam())

        # Compute storevars and external modvars
        E_K = (((R*T)/F)*(torch.log((p.Ko/self.sv[:, self.si.Ki]))))
        ICaL = (((V_row[:, self.V_ti.vrow_29_idx]*self.sv[:, self.si.d])*self.sv[:, self.si.f])*self.sv[:, self.si.f_Ca])
        IKACh = (((p.GACh*(10.0/(1.0+(9.13652/(pow(p.ACh,0.477811))))))*(0.0517+(0.4516/(1.0+(torch.exp(((V+59.53)/17.18))))))) * (V - E_K))
        INa = (((((V_row[:, self.V_ti.vrow_7_idx]*self.sv[:, self.si.m])*self.sv[:, self.si.m])*self.sv[:, self.si.m])*self.sv[:, self.si.h])*self.sv[:, self.si.j])
        INaCa = (V_row[:, self.V_ti.vrow_31_idx] - (self.sv[:, self.si.Cai] * V_row[:, self.V_ti.vrow_32_idx]))
        vrow_13 = (p.Gto * (V - E_K))
        vrow_18 = ((p.factorGKur*V_row[:, self.V_ti.GKur_idx]) * (V - E_K))
        vrow_21 = ((p.GKr * (V - E_K)) / (1. + (torch.exp(((V + 15.) / 22.4)))))
        vrow_24 = (p.GKs * (V - E_K))
        vrow_8 = ((p.GK1 * (V - E_K)) / (1. + (torch.exp((0.07 * (V + 80.))))))
        IK1 = vrow_8
        IKr = (vrow_21*self.sv[:, self.si.xr])
        IKs = ((vrow_24*self.sv[:, self.si.xs])*self.sv[:, self.si.xs])
        IKur = ((((vrow_18*self.sv[:, self.si.ua])*self.sv[:, self.si.ua])*self.sv[:, self.si.ua])*self.sv[:, self.si.ui])
        IpCa = (Cai_row[:, self.Cai_ti.carow_3_idx]+V_row[:, self.V_ti.vrow_36_idx])
        Ito = ((((vrow_13*self.sv[:, self.si.oa])*self.sv[:, self.si.oa])*self.sv[:, self.si.oa])*self.sv[:, self.si.oi])
        Iion = INa+IK1+Ito+IKur+IKr+IKs+ICaL+IpCa+INaCa+V_row[:, self.V_ti.IbNa_idx]+V_row[:, self.V_ti.INaK_idx]+IKACh

        # Complete Forward Euler Update
        Itr = ((p.factorGtr*(self.sv[:, self.si.Ca_up]-(self.sv[:, self.si.Ca_rel])))/tau_tr)
        Irel = ((((((p.factorGrel*self.sv[:, self.si.u])*self.sv[:, self.si.u])*self.sv[:, self.si.v])*k_rel)*self.sv[:, self.si.w])*(self.sv[:, self.si.Ca_rel]-(Cai_row[:, self.Cai_ti.conCa_idx])))
        dIups = (Cai_row[:, self.Cai_ti.carow_1_idx] - ((p.maxIup / p.maxCaup) * self.sv[:, self.si.Ca_up]))
        diff_Ca_rel = ((Itr - Irel) / (1. + ((C_dCa_rel / (self.sv[:, self.si.Ca_rel] + KmCsqn)) / (self.sv[:, self.si.Ca_rel] + KmCsqn))))
        diff_Ki = ((-((((((Ito+IKr)+IKur)+IKs)+IK1)+IKACh) - (2.0 * V_row[:, self.V_ti.INaK_idx]))) / (F * Voli))
        diff_Ca_up = (dIups - (Itr * C_dCaup))
        diff_Cai = ((((C_B1d * (((INaCa+INaCa) - IpCa) - ICaL)) - (C_B1e * dIups)) + Irel) / Cai_row[:, self.Cai_ti.carow_2_idx])

        self.sv[:, self.si.Ca_rel] += diff_Ca_rel * self.dt
        self.sv[:, self.si.Ca_up] += diff_Ca_up * self.dt
        self.sv[:, self.si.Cai] += diff_Cai * self.dt
        self.sv[:, self.si.Ki] += diff_Ki * self.dt

        # Complete Rush Larsen Update
        fn = ((C_Fn1*Irel) - (C_Fn2 * (ICaL - (0.4 * INaCa))))
        fn_row = interpolate(fn, self.fn_tab, fn_TableParam())

        self.rush_larsen_B[:, [0, 1, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14]] = V_row[:, [self.V_ti.d_rush_larsen_B_idx,
                                                                                    self.V_ti.f_rush_larsen_B_idx,
                                                                                    self.V_ti.h_rush_larsen_B_idx,
                                                                                    self.V_ti.j_rush_larsen_B_idx,
                                                                                    self.V_ti.m_rush_larsen_B_idx,
                                                                                    self.V_ti.oa_rush_larsen_B_idx,
                                                                                    self.V_ti.oi_rush_larsen_B_idx,
                                                                                    self.V_ti.ua_rush_larsen_B_idx,
                                                                                    self.V_ti.ui_rush_larsen_B_idx,
                                                                                    self.V_ti.w_rush_larsen_B_idx,
                                                                                    self.V_ti.xr_rush_larsen_B_idx,
                                                                                    self.V_ti.xs_rush_larsen_B_idx]]
        self.rush_larsen_B[:, 11] = fn_row[:, self.fn_ti.v_rush_larsen_B_idx]

        self.rush_larsen_A[:, [0, 1, 3, 4, 5, 6, 7, 9, 10, 12, 13, 14]] = V_row[:, [self.V_ti.d_rush_larsen_A_idx,
                                                                                    self.V_ti.f_rush_larsen_A_idx,
                                                                                    self.V_ti.h_rush_larsen_A_idx,
                                                                                    self.V_ti.j_rush_larsen_A_idx,
                                                                                    self.V_ti.m_rush_larsen_A_idx,
                                                                                    self.V_ti.oa_rush_larsen_A_idx,
                                                                                    self.V_ti.oi_rush_larsen_A_idx,
                                                                                    self.V_ti.ua_rush_larsen_A_idx,
                                                                                    self.V_ti.ui_rush_larsen_A_idx,
                                                                                    self.V_ti.w_rush_larsen_A_idx,
                                                                                    self.V_ti.xr_rush_larsen_A_idx,
                                                                                    self.V_ti.xs_rush_larsen_A_idx]]
        self.rush_larsen_A[:, 2] = Cai_row[:, self.Cai_ti.f_Ca_rush_larsen_A_idx]
        self.rush_larsen_A[:, 8] = fn_row[:, self.fn_ti.u_rush_larsen_A_idx]
        self.rush_larsen_A[:, 11] = fn_row[:, self.fn_ti.v_rush_larsen_A_idx]

        self.sv[:, 4:] = self.rush_larsen_A + self.rush_larsen_B * self.sv[:, 4:]

        return -Iion



if __name__ == "__main__":
    n_nodes = 100
    dt = 0.001
    device = "cuda"
    cm = Courtemanche(dt, device)
    cm.construct_tables()
    V = cm.initialize(n_nodes)
    du = cm.differentiate(V)
    print(du)