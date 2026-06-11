import warnings
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import time
import torch
import torchcor as tc
from pathlib import Path
from torchcor.core.mesh import MeshReader
from torchcor.core.stimulation import Stimuli
from torchcor.simulator.monodomain import Monodomain


# A large but finite "infinity" for not-yet-reached arrival times. Keeping it
# finite avoids nan/inf propagation in the vectorised local solvers below.
_INF = 1.0e9
_INF2 = 5.0e8


# --------------------------------------------------------------------------- #
#  Anisotropic eikonal model (fast iterative method)                          #
# --------------------------------------------------------------------------- #
#  The wavefront arrival times t_a are the viscosity solution of the
#  anisotropic eikonal equation (Neic et al. 2017, eq. 24-25)
#
#       sqrt( grad(t_a)^T  V  grad(t_a) ) = 1            in  Omega
#                                   t_a   = t_0          on  Gamma
#
#  with the squared-velocity tensor  V = v_l ll^T + v_t tt^T + v_n nn^T, where
#  v_l = cv_l^2 and v_t = v_n = cv_t^2 are the squared conduction velocities
#  along / across the fibre direction l.  The travel time of a displacement d is
#  ||d||_M = sqrt(d^T M d) with the metric M = V^{-1},
#
#       M = (1/cv_t^2) I + (1/cv_l^2 - 1/cv_t^2) l l^T .
#
#  It is solved with a Jacobi flavour of the Fast Iterative Method: every simplex
#  provides a monotone, consistent local update for each of its vertices and the
#  field is relaxed to convergence.  The local solvers are the standard simplex
#  updates -- a vertex is updated from the opposite facet:
#     * triangle (2-simplex):  vertex <- opposite edge   (1-D minimisation)
#     * tetrahedron (3-simplex): vertex <- opposite face  (2-D minimisation,
#       whose boundary reduces to the edge updates of that face)
#     * line (1-simplex):      vertex <- the other vertex (anisotropic length)
#  Every update is vectorised over the whole mesh so the solve runs on the GPU.
# --------------------------------------------------------------------------- #

# (target, p, q): for each tet vertex, the three edges of its opposite face.
_TET_EDGES = [(0, 1, 2), (0, 2, 3), (0, 3, 1),
              (1, 2, 3), (1, 3, 0), (1, 0, 2),
              (2, 3, 0), (2, 0, 1), (2, 1, 3),
              (3, 0, 1), (3, 1, 2), (3, 2, 0)]
# (target, s0, s1, s2): each tet vertex and its opposite face.
_TET_FACES = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3), (3, 0, 1, 2)]
# (target, p, q): each triangle vertex and its opposite edge.
_TRI_EDGES = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]


def build_metric(fibres, cv_l, cv_t, device, dtype):
    """Per-element eikonal metric M = V^{-1} from conduction velocities.

    fibres : (E, 3) unit fibre vectors
    cv_l   : (E,)   longitudinal conduction velocity (mm/ms)
    cv_t   : (E,)   transverse conduction velocity   (mm/ms)
    """
    E = fibres.shape[0]
    eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(E, 3, 3)
    ff = fibres.unsqueeze(2) @ fibres.unsqueeze(1)          # (E, 3, 3)

    inv_l2 = (1.0 / (cv_l * cv_l)).view(E, 1, 1)
    inv_t2 = (1.0 / (cv_t * cv_t)).view(E, 1, 1)
    return inv_t2 * eye + (inv_l2 - inv_t2) * ff


def region_velocities(regions, velocity_map, device, dtype):
    """Expand a {region_id: (cv_l, cv_t)} map onto a per-element array."""
    cv_l = torch.zeros(regions.shape[0], device=device, dtype=dtype)
    cv_t = torch.zeros(regions.shape[0], device=device, dtype=dtype)

    covered = torch.zeros(regions.shape[0], device=device, dtype=torch.bool)
    for rid, (vl, vt) in velocity_map.items():
        mask = regions == rid
        cv_l[mask] = vl
        cv_t[mask] = vt
        covered |= mask

    if not bool(covered.all()):
        missing = torch.unique(regions[~covered]).tolist()
        raise Exception(f"No conduction velocity specified for region(s) {missing}. "
                        f"Call add_velocity(...) for every region.")
    return cv_l, cv_t


def _quad(u, metric, v):
    """Batched metric contraction u^T M v for (E,3) vectors and (E,3,3) M."""
    return torch.einsum('ei,eij,ej->e', u, metric, v)


def _edge_terms(nodes, tgt, p, q, metric):
    """Geometry-only constants of the 1-D update of `tgt` from the segment p-q.

    The update minimises, over P = X_p + xi (X_q - X_p), xi in [0, 1],
        T_tgt = T_p + xi (T_q - T_p) + ||X_tgt - P||_M ,
    which only needs a = e^T M e, b = e^T M w, c = w^T M w (e = X_q-X_p,
    w = X_tgt-X_p) and the two endpoint travel times.
    """
    e = nodes[q] - nodes[p]
    w = nodes[tgt] - nodes[p]
    a = _quad(e, metric, e)
    b = _quad(e, metric, w)
    c = _quad(w, metric, w)
    dpt = torch.sqrt(torch.clamp(c, min=0.0))                 # travel time p -> tgt
    dqt = torch.sqrt(torch.clamp(a - 2.0 * b + c, min=0.0))   # travel time q -> tgt
    return torch.stack([tgt, p, q]), torch.stack([a, b, c, dpt, dqt])


def _face_terms(nodes, tgt, s0, s1, s2, metric):
    """Geometry-only constants of the 2-D update of `tgt` from the face s0-s1-s2.

    The interior update minimises, over P = X0 + xi e1 + eta e2 in the face,
        T_tgt = T0 + xi (T1-T0) + eta (T2-T0) + ||X_tgt - P||_M ,
    with e1 = X1-X0, e2 = X2-X0, w = X_tgt-X0.  Stored are the metric Gram
    entries a11,a12,a22, the projections b1,b2, c0 = w^T M w and det.
    """
    e1 = nodes[s1] - nodes[s0]
    e2 = nodes[s2] - nodes[s0]
    w = nodes[tgt] - nodes[s0]
    a11 = _quad(e1, metric, e1)
    a12 = _quad(e1, metric, e2)
    a22 = _quad(e2, metric, e2)
    b1 = _quad(e1, metric, w)
    b2 = _quad(e2, metric, w)
    c0 = _quad(w, metric, w)
    det = a11 * a22 - a12 * a12
    return torch.stack([tgt, s0, s1, s2]), torch.stack([a11, a12, a22, b1, b2, c0, det])


def build_ops(nodes, elems, velocity_map, fibres, device, dtype):
    """Pre-compute every local-update operator of the mesh, once.

    Returns (edge, face): `edge` drives the 1-D edge/segment updates (triangles,
    tet face-edges, lines), `face` the 2-D tet interior updates.  Each is a pair
    (idx, coef) of stacked per-update index and constant tensors, or None.
    """
    edge_idx, edge_coef, face_idx, face_coef = [], [], [], []

    def add_edges(conn, metric, pattern):
        for t, p, q in pattern:
            idx, coef = _edge_terms(nodes, conn[:, t], conn[:, p], conn[:, q], metric)
            edge_idx.append(idx)
            edge_coef.append(coef)

    if elems.Tr.data is not None:
        cv_l, cv_t = region_velocities(elems.Tr.region, velocity_map, device, dtype)
        add_edges(elems.Tr.data, build_metric(fibres[elems.Tr.idx], cv_l, cv_t, device, dtype), _TRI_EDGES)

    if elems.Tt.data is not None:
        cv_l, cv_t = region_velocities(elems.Tt.region, velocity_map, device, dtype)
        tet = elems.Tt.data
        metric = build_metric(fibres[elems.Tt.idx], cv_l, cv_t, device, dtype)
        add_edges(tet, metric, _TET_EDGES)
        for t, s0, s1, s2 in _TET_FACES:
            idx, coef = _face_terms(nodes, tet[:, t], tet[:, s0], tet[:, s1], tet[:, s2], metric)
            face_idx.append(idx)
            face_coef.append(coef)

    if elems.Ln.data is not None:
        cv_l, cv_t = region_velocities(elems.Ln.region, velocity_map, device, dtype)
        # propagation along a cable is isotropic at cv_l; degenerate source p==q
        metric = build_metric(fibres[elems.Ln.idx], cv_l, cv_l, device, dtype)
        add_edges(elems.Ln.data, metric, [(0, 1, 1), (1, 0, 0)])

    if not edge_idx and not face_idx:
        raise Exception("No elements found to build the eikonal operator.")

    edge = (torch.cat(edge_idx, dim=1), torch.cat(edge_coef, dim=1)) if edge_idx else None
    face = (torch.cat(face_idx, dim=1), torch.cat(face_coef, dim=1)) if face_idx else None
    return edge, face


def _relax_edges(out, T, edge):
    """Scatter the best 1-D edge update into `out` (one Jacobi pass)."""
    (tgt, p, q), (a, b, c, dpt, dqt) = edge
    tp, tq = T[p], T[q]
    both = (tp < _INF2) & (tq < _INF2)

    # endpoint (single-vertex) candidates
    cand = torch.minimum(torch.where(tp < _INF2, tp + dpt, _INF),
                         torch.where(tq < _INF2, tq + dqt, _INF))

    # interior critical point of the (convex) cost; its stationarity squares to
    #   a(a - u^2) xi^2 - 2 b(a - u^2) xi + (b^2 - u^2 c) = 0
    u = tq - tp
    au = a - u * u
    A = a * au
    B = -2.0 * b * au
    C = b * b - u * u * c
    sq = torch.sqrt(torch.clamp(B * B - 4.0 * A * C, min=0.0))
    safe_A = torch.where(A.abs() > 1e-30, A, 1.0)
    for xi in ((-B + sq) / (2.0 * safe_A),
               (-B - sq) / (2.0 * safe_A),
               b / torch.where(a > 1e-30, a, 1.0)):                  # last: u -> 0
        valid = both & (xi >= 0.0) & (xi <= 1.0)
        nrm = torch.sqrt(torch.clamp(a * xi * xi - 2.0 * b * xi + c, min=0.0))
        cand = torch.minimum(cand, torch.where(valid, tp + xi * u + nrm, _INF))

    out.scatter_reduce_(0, tgt, cand, reduce='amin', include_self=True)


def _relax_faces(out, T, face, n_fixed_point=8):
    """Scatter the best 2-D tetrahedral-face interior update into `out`.

    The stationarity conditions are linear in (xi, eta) for a fixed travel time
    n = ||X_tgt - P||_M, so a short fixed-point alternates a 2x2 solve with a
    refresh of n.  Only points strictly inside the face are kept; the face
    boundary is already covered by the edge updates, and any non-converged point
    is simply a valid (if looser) upper bound, so the scheme stays monotone.
    """
    (tgt, s0, s1, s2), (a11, a12, a22, b1, b2, c0, det) = face
    t0, t1, t2 = T[s0], T[s1], T[s2]
    allf = (t0 < _INF2) & (t1 < _INF2) & (t2 < _INF2)
    u1, u2 = t1 - t0, t2 - t0

    safe_det = torch.where(det.abs() > 1e-30, det, 1.0)
    n = torch.sqrt(torch.clamp(c0, min=0.0))                  # initial guess: dist(s0 -> tgt)
    xi = torch.zeros_like(n)
    eta = torch.zeros_like(n)
    for _ in range(n_fixed_point):
        r1, r2 = b1 - u1 * n, b2 - u2 * n
        xi = (r1 * a22 - r2 * a12) / safe_det
        eta = (a11 * r2 - a12 * r1) / safe_det
        n = torch.sqrt(torch.clamp(a11 * xi * xi + 2.0 * a12 * xi * eta + a22 * eta * eta
                                   - 2.0 * b1 * xi - 2.0 * b2 * eta + c0, min=0.0))

    inside = allf & (det > 1e-30) & (xi >= 0.0) & (eta >= 0.0) & (xi + eta <= 1.0)
    cand = torch.where(inside, t0 + xi * u1 + eta * u2 + n, _INF)
    out.scatter_reduce_(0, tgt, cand, reduce='amin', include_self=True)


def fim_eikonal(edge, face, seed_time, tol=1e-3, max_iter=100000, verbose=True):
    """Fast Iterative Method for the anisotropic eikonal equation."""
    T = seed_time.clone()
    start = time.time()
    n_iter = 0
    for n in range(1, max_iter + 1):
        out = T.clone()
        if edge is not None:
            _relax_edges(out, T, edge)
        if face is not None:
            _relax_faces(out, T, face)
        out = torch.minimum(out, seed_time)                  # keep seeds pinned
        change = (T - out).abs().max().item()
        T = out
        n_iter = n
        if change < tol:
            break

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if verbose:
        reached = int((T < _INF2).sum().item())
        print(f"eikonal: {n_iter} sweeps | {reached}/{T.numel()} nodes activated | "
              f"{time.time() - start:.2f}s", flush=True)

    return torch.where(T < _INF2, T, torch.full_like(T, float('nan')))


# --------------------------------------------------------------------------- #
#  Stand-alone eikonal simulator                                              #
# --------------------------------------------------------------------------- #
class Eikonal:
    """Anisotropic eikonal activation-time solver with a Monodomain-like API."""

    def __init__(self, device=None, dtype=None):
        self.device = tc.get_device() if device is None else device
        self.dtype = torch.float64 if dtype is None else dtype

        self.n_nodes = None
        self.nodes = None
        self.elems = None
        self.fibres = None
        self.regions = None
        self.unique_regions = None

        self._vel = {}
        self.seed_time = None
        self.eikonal_AT = None
        self.mesh_path = None

    def load_mesh(self, path="Data/atrium/Case_1", unit_conversion=1000):
        self.mesh_path = Path(path)
        nodes, elems, regions, fibres = MeshReader(path).read(unit_conversion=unit_conversion)

        self.n_nodes = nodes.shape[0]
        self.nodes = torch.from_numpy(nodes).to(dtype=self.dtype, device=self.device)
        self.elems = elems.to_torch(self.device)
        self.regions = torch.from_numpy(regions).to(dtype=torch.long, device=self.device)
        self.unique_regions = torch.unique(self.regions).tolist()
        self.fibres = torch.from_numpy(fibres).to(dtype=self.dtype, device=self.device)

        self.seed_time = torch.full((self.n_nodes,), _INF, device=self.device, dtype=self.dtype)

    def add_velocity(self, region_ids, vel_l, vel_t):
        """Conduction velocities (mm/ms, i.e. numerically m/s) per region."""
        if region_ids is None:
            region_ids = self.unique_regions
        for rid in region_ids:
            self._vel[rid] = (float(vel_l), float(vel_t))

    def add_stimulus(self, vtx_filepath, start=0.0):
        """Seed the wavefront from the nodes listed in a .vtx file at time `start`."""
        region = Stimuli(self.n_nodes, self.device, self.dtype).load_stimulus_region(vtx_filepath)
        self.seed_time[region] = torch.minimum(self.seed_time[region],
                                               torch.full_like(self.seed_time[region], float(start)))

    def solve(self, tol=1e-3, max_iter=100000, verbose=True):
        edge, face = build_ops(self.nodes, self.elems, self._vel, self.fibres, self.device, self.dtype)
        self.eikonal_AT = fim_eikonal(edge, face, self.seed_time, tol=tol, max_iter=max_iter, verbose=verbose)
        return self.eikonal_AT


# --------------------------------------------------------------------------- #
#  Reaction-Eikonal simulator                                                 #
# --------------------------------------------------------------------------- #
class ReactionEikonal(Monodomain):
    """Reaction-Eikonal model (Neic et al., J. Comput. Phys. 2017).

    The eikonal model supplies the activation-time field t_a(x); the full ionic
    reaction is then recovered locally by triggering each cell with a current
    centred on its activation time.  Two trigger currents are supported:

      * default: an action-potential foot current  (A_F / tau_F) * exp((t - t_a) / tau_F)
        in the window [t_a, t_a + T_foot], switched off once Vm reaches V_th.  A_F
        is self-calibrated to each cell's rest->V_th gap, so it works with any
        ionic model out of the box.
      * opt-in via `set_diffusion_current(...)`: a triple-Gaussian diffusion
        current I_diff that approximates div(sigma grad Vm) and must be fit once
        per ionic model.

    The `diffusion` flag selects the model variant:

      * diffusion=False  ->  R-E (eq. 30): NO diffusion term, so every node is an
        independent ODE  Cm dVm/dt = I_foot - I_ion, the foot firing each cell at
        its own t_a.  Cells are electrically decoupled (no current flows between
        neighbours) and there is no linear solve, so it is fast.  Needs only
        add_velocity.  Use it for activation/repolarisation maps and fast runs.
      * diffusion=True   ->  R-E+ (eq. 32): ADDS the monodomain diffusion operator,
            Cm dVm/dt = I_foot - I_ion + div(sigma grad Vm)/beta,
        solved with conjugate gradient every step.  This couples neighbouring
        cells, recovering electrotonic loading / source-sink effects -- needed for
        accurate electrograms / ECGs.  It is slower (a linear solve per step) and
        also needs add_conductivity, calibrated to the same CV as add_velocity.

    Mesh handling, ionic models, FEM assembly, activation / repolarisation maps
    and VTK/IGB export are inherited unchanged from Monodomain, so the interface
    matches the monodomain solver.

    Re-stimulation reuses the seed stimulus' period/count, i.e. each beat repeats
    the same activation sequence; dynamic restitution / reentry is out of scope.
    """

    def __init__(self, ionic_models, T, dt, diffusion=False,
                 v_th=-50.0, t_foot=5.0, tau_foot=0.7,
                 device=None, dtype=None, mass_lumping=False):
        super().__init__(ionic_models, T, dt, device=device, dtype=dtype, mass_lumping=mass_lumping)

        self.diffusion = diffusion          # False = R-E (no diffusion); True = R-E+ (with diffusion solve)
        self.v_th = v_th                    # foot cut-off voltage V_th (mV)
        self.t_foot = t_foot                # foot window length T_foot (ms)
        self.tau_foot = tau_foot            # foot time constant tau_F (ms)

        self._vel = {}
        self._gauss = None                  # triple-Gaussian diffusion current, if set
        self.eikonal_AT = None
        self.foot_amp = None
        self.period = T
        self.count = 1

    def add_velocity(self, region_ids, vel_l, vel_t):
        """Conduction velocities (mm/ms == m/s) per region for the eikonal solve."""
        if region_ids is None:
            region_ids = self.unique_regions
        for rid in region_ids:
            self._vel[rid] = (float(vel_l), float(vel_t))

    def set_diffusion_current(self, alpha, beta, gamma):
        """Use a triple-Gaussian diffusion current instead of the default AP foot.

        Replaces the trigger by
            I_diff(s) = sum_i alpha_i * exp(-((s - beta_i)/gamma_i)^2),   s = t - t_a,
        which approximates div(sigma grad Vm).  `alpha` (mV/ms), `beta` (ms) and
        `gamma` (ms) are length-3 vectors fit once to a 1-D monodomain upstroke
        of the chosen ionic model.  Applied within +/- T_foot of each arrival.
        """
        tensor = lambda v: torch.as_tensor(v, device=self.device, dtype=self.dtype)
        self._gauss = (tensor(alpha), tensor(beta), tensor(gamma))

    # ----- eikonal stage ----- #
    def eikonal_activation_times(self, tol=1e-3, max_iter=100000, verbose=True):
        """Solve the eikonal equation for the activation-time field t_a(x).

        The fast standalone step: build the local-update operators from the mesh
        and conduction velocities, seed the front from the added stimuli, and run
        the FIM -- no reaction and no linear solve.  The field is cached in
        `self.eikonal_AT` (and reused by `solve()`) and returned; on its own it is
        already the activation map.  Requires load_mesh / add_velocity / add_stimulus.
        """
        seed_time = torch.full((self.n_nodes,), _INF, device=self.device, dtype=self.dtype)
        for stim in self.stimuli.stimulus_list:
            mask = stim.stimulus != 0
            seed_time[mask] = torch.minimum(seed_time[mask],
                                            torch.full_like(seed_time[mask], float(stim.start)))
        if bool((seed_time >= _INF2).all()):
            raise Exception("No stimulus added: the eikonal model has no seed nodes.")

        edge, face = build_ops(self.nodes, self.elems, self._vel, self.fibres, self.device, self.dtype)
        self.eikonal_AT = fim_eikonal(edge, face, seed_time, tol=tol, max_iter=max_iter, verbose=verbose)
        return self.eikonal_AT

    # ----- trigger current (couples eikonal -> reaction) ----- #
    def _foot_current(self, t, u):
        """Eikonal-triggered current at offset s = t - t_a.

        Active in the post-arrival window s in [0, T_foot] of each (possibly paced)
        arrival; comparisons against the nan of never-activated nodes are False, so
        those nodes stay inactive.

        Default (foot): the foot current  (A_F / tau_F) * exp(s / tau_F), gated off
        once Vm reaches V_th, so a resting cell is ramped up to threshold just after
        t_a and the intrinsic upstroke takes over.  This also triggers the seed
        nodes (t_a = stimulus onset), so no separate stimulus is injected.

        Triple-Gaussian variant (if set_diffusion_current was called): I_diff.
        """
        rel = t - self.eikonal_AT
        beat = torch.clamp(torch.floor(rel / self.period), min=0.0, max=self.count - 1)
        s = rel - beat * self.period                        # time since latest arrival (>= 0 in window)
        in_window = (s >= 0.0) & (s <= self.t_foot)

        if self._gauss is None:
            active = in_window & (u < self.v_th)
            rate = (self.foot_amp / self.tau_foot) * torch.exp(s / self.tau_foot)
        else:
            alpha, beta, gamma = self._gauss                # length-3 each
            z = (s.unsqueeze(-1) - beta) / gamma            # (N, 3)
            rate = (alpha * torch.exp(-z * z)).sum(dim=-1)  # (N,)
            active = in_window
        return torch.where(active, rate, 0.0)

    # ----- one time step ----- #
    def step(self, u, t, a_tol, r_tol, max_iter, verbose=False):
        if verbose and torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        ### ionic ###
        b = u.clone()
        for im in self.ionic_models:
            idx = im.node_indices
            du = im.differentiate(u[idx]) / 100
            b[idx] = u[idx] * self.Cm + self.dt * du

        ### eikonal-triggered current (also fires the seed nodes) ###
        b += self.dt * self._foot_current(t, u) / 100

        if verbose and torch.cuda.is_available():
            torch.cuda.synchronize()
        ionic_time = time.time() - start_time
        start_time = time.time()

        ### electric ###
        if self.diffusion:
            # R-E+: add the monodomain diffusion operator and solve A u = b with CG
            b = self.M @ b
            b -= (1 - self.theta) * self.dt * self.K @ u
            u, n_iter = self.cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)
        else:
            # R-E: no diffusion -> A = Cm I, an explicit per-node ODE update (no solve)
            u = b / self.Cm
            n_iter = 0

        if verbose and torch.cuda.is_available():
            torch.cuda.synchronize()
        electric_time = time.time() - start_time
        return u, n_iter, ionic_time, electric_time

    # ----- driver ----- #
    def solve(self, a_tol=1e-5, r_tol=1e-5, max_iter=100, linear_guess=True,
              snapshot_interval=5, verbose=True, result_path=None,
              eikonal_tol=1e-3, eikonal_max_iter=100000):
        self.result_path = Path(result_path) if result_path is not None else None
        self.snapshot_interval = snapshot_interval

        # 1. eikonal activation times (reuse if already computed standalone)
        if self.eikonal_AT is None:
            self.eikonal_activation_times(tol=eikonal_tol, max_iter=eikonal_max_iter, verbose=verbose)

        # 2. FEM operators only needed for the R-E+ diffusion term
        if self.diffusion:
            self.assemble()

        # 3. initial state from the ionic models
        u = torch.zeros((self.n_nodes), dtype=self.dtype, device=self.device)
        for im in self.ionic_models:
            u[im.node_indices] = im.initialize(im.node_indices.shape[0]).clone()
        u_initial = u.clone()

        # 4. foot-current amplitude A_F, self-calibrated to each cell's rest->V_th gap
        self.foot_amp = self.v_th - u_initial
        if len(self.stimuli.stimulus_list) > 0:
            self.period = self.stimuli.stimulus_list[0].period
            self.count = self.stimuli.stimulus_list[0].count

        if self.diffusion:
            self.cg.initialize(x=u, linear_guess=linear_guess)
        ts_per_frame = int(snapshot_interval / self.dt)

        t = 0.0
        solving_time = time.time()
        total_ionic_time = 0.0
        total_electric_time = 0.0
        n_total_iter = 0
        solution_list = [u_initial]
        for n in range(1, self.nt + 1):
            t += self.dt

            u, n_iter, ionic_time, electric_time = self.step(u, t, a_tol, r_tol, max_iter, verbose)
            if self.diffusion and n_iter >= max_iter:
                raise Exception("exceeded max_iter")

            n_total_iter += n_iter
            total_ionic_time += ionic_time
            total_electric_time += electric_time

            if n % ts_per_frame == 0:
                solution_list.append(u.clone())
                if verbose and snapshot_interval != self.T:
                    print(f"t: {round(t, 1)}/{self.T} |",
                          f"Time elapsed: {round(time.time() - solving_time, 1)} |",
                          f"CG iter:", n_total_iter, flush=True)
                n_total_iter = 0

        if verbose:
            model = "R-E+" if self.diffusion else "R-E"
            print(f"[{model}] nodes: {self.n_nodes} | "
                  f"total_time: {time.time() - solving_time:.2f}s | "
                  f"ionic_time: {total_ionic_time:.2f}s | "
                  f"electric_time: {total_electric_time:.2f}s", flush=True)

        return torch.stack(solution_list, dim=0)
