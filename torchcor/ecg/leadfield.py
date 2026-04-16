"""
ECG Lead Field Solver -- Reciprocal Method (openCARP-compatible)

Computes body-surface ECGs from transmembrane potential (Vm) using the
reciprocal (adjoint) lead field method on a combined heart-torso FE mesh.

Physics (Potse 2018, Bishop & Plank 2011):

    Bidomain elliptic equation (phi_e recovery):

        div( (sigma_i + sigma_e) grad(phi_e) ) = -div( sigma_i  grad(Vm) )

    On the full torso domain the bulk conductivity is:

        sigma_bulk = sigma_i + sigma_e   in the myocardium
        sigma_bulk = sigma_T             in extracardiac tissue

    Heart-torso coupling (Krassowska & Neu 1994):
        - Potential continuity:  phi_e = u_T  on the epicardial surface
        - Current continuity:    sigma_e grad(phi_e).n = sigma_T grad(u_T).n
        Both are satisfied naturally by FEM on the combined mesh.

Reciprocal method:

    For each electrode e with ground g, solve ONCE:

        K  Z_e  =  delta_e  -  delta_g          (1)

    where K is the stiffness matrix with sigma_bulk.
    The system is pure Neumann (insulating body surface) so the
    stiffness matrix is singular (constant null space).  A penalty
    term pins Z(g) ~ 0 to lift the singularity while preserving
    the Neumann physics everywhere else.

    The lead field vector is:

        q_e  =  -(K_i  Z_e)                     (2)

    where K_i is assembled from sigma_i on heart elements only
    (in torso-mesh node numbering).

    The ECG signal at every time step is then:

        V_e(t)  =  Vm(t)  @  q_e                (3)

    where Vm is (T, N_heart) and q_e is (N_heart,).

Implementation notes:
    - Both K and K_i are assembled on the torso mesh using the torso
      mesh's own fibre directions, so there is no element-ordering or
      node-numbering mismatch.
    - The CG solve runs in float64 (matching openCARP / PETSc default)
      for full numerical accuracy on large meshes.
    - Neumann BC with penalty grounding: current is injected at the
      electrode (+I) and extracted at the ground (-I), with a penalty
      term alpha * e_g * e_g^T added to pin Z(g) ~ 0.
"""

from typing import Dict, Optional
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from torchcor.core import (
    MeshReader,
    Matrices3D,
    Preconditioner,
    ConjugateGradient,
    Conductivity,
)

import warnings
warnings.filterwarnings(
    "ignore", message="Sparse CSR tensor support is in beta state"
)

Tensor = torch.Tensor


# ====================================================================== #
#  LeadField class
# ====================================================================== #
class LeadField:
    """
    ECG lead-field solver (reciprocal method, openCARP-compatible).

    All stiffness matrices live on the torso mesh.  The standalone heart
    mesh is loaded only for the Vm node-index mapping.
    """

    def __init__(self, torso_mesh_dir, heart_mesh_dir, device, dtype):
        self.device = device
        self.dtype = dtype                       # storage / Vm dtype

        # ---- meshes ----
        self._load_torso_mesh(torso_mesh_dir)
        self._load_heart_mesh(heart_mesh_dir)

        # ---- per-element sigma for the torso mesh (set via add_*) ----
        self.torso_sigma = torch.zeros(
            (self.torso_regions.shape[0], 3, 3),
            device=device, dtype=dtype,
        )
        self.n_torso = int(self.torso_nodes.shape[0])

        # ---- heart conductivity bookkeeping ----
        self._heart_cond_params = []
        self.heart_tags = []

        # ---- electrodes ----
        self.electrodes: Dict[str, int] = {}
        self.ground = "RL"

        # ---- matrices (filled by build()) ----
        self.K_torso = None          # (sigma_i+sigma_e) heart, sigma_T torso
        self.K_i = None              # sigma_i on heart elems, torso numbering

        # ---- mapping (filled by build()) ----
        self.heart_to_torso = None   # (N_heart,) int64

        # ---- lead fields (filled by precompute_*) ----
        self.q_heart: Dict[str, Tensor] = {}

        # ---- solver cache (filled by _prepare_solver()) ----
        self._A = None               # float64 CSR: K_torso + penalty
        self._K_i_f64 = None         # float64 CSR of K_i
        self._pcd = None             # Jacobi preconditioner (float64)

    # ================================================================== #
    #  Mesh I/O
    # ================================================================== #
    def _load_torso_mesh(self, mesh_dir, unit_conversion=1000):
        reader = MeshReader(mesh_dir)
        nodes, elems, _, fibres = reader.read(unit_conversion=unit_conversion)

        self.torso_nodes   = torch.from_numpy(nodes).to(self.device, self.dtype)
        self.torso_elems   = torch.from_numpy(elems.Tt.data).to(self.device, torch.long)
        self.torso_regions = torch.from_numpy(elems.Tt.region).to(self.device, torch.long)
        self.torso_fibres  = torch.from_numpy(
            fibres[elems.Tt.idx]
        ).to(self.device, self.dtype)

    def _load_heart_mesh(self, mesh_dir, unit_conversion=1000):
        reader = MeshReader(mesh_dir)
        nodes, elems, _, fibres = reader.read(unit_conversion=unit_conversion)

        self.heart_nodes   = torch.from_numpy(nodes).to(self.device, self.dtype)
        self.heart_elems   = torch.from_numpy(elems.Tt.data).to(self.device, torch.long)
        self.heart_regions = torch.from_numpy(elems.Tt.region).to(self.device, torch.long)
        self.heart_fibres  = torch.from_numpy(
            fibres[elems.Tt.idx]
        ).to(self.device, self.dtype)

    # ================================================================== #
    #  Conductivity registration
    # ================================================================== #
    def add_torso_conductivity(self, tags, g):
        """Isotropic scalar conductivity *g* (S/m) for torso region tags."""
        I3 = torch.eye(3, device=self.device, dtype=self.dtype)
        for tag in tags:
            mask = self.torso_regions == int(tag)
            self.torso_sigma[mask] = float(g) * I3

    def add_heart_conductivity(self, region_ids, il, it, el=None, et=None):
        """Anisotropic bidomain conductivity for heart region tags."""
        self.heart_tags.extend(region_ids)
        self._heart_cond_params.append((region_ids, il, it, el, et))

    # ================================================================== #
    #  FEM assembly
    # ================================================================== #
    def build(self):
        """
        Assemble the two stiffness matrices on the torso mesh.

        K_torso  (N_torso x N_torso)
            Bulk conductivity: sigma_i + sigma_e on heart elements,
            sigma_T on extracardiac elements.

        K_i  (N_torso x N_torso, very sparse)
            Intracellular conductivity sigma_i on heart elements only.
            Non-heart rows/cols are structurally zero.
        """
        tag_t = torch.tensor(self.heart_tags, device=self.device, dtype=torch.long)
        heart_mask = torch.isin(self.torso_regions, tag_t)

        # -- heart conductivity from TORSO-mesh fibres ------------------
        torso_heart_regions = self.torso_regions[heart_mask]
        torso_heart_fibres  = self.torso_fibres[heart_mask]

        cond = Conductivity(torso_heart_regions, dtype=self.dtype)
        for region_ids, il, it, el, et in self._heart_cond_params:
            cond.add(region_ids, il, it, el, et)
        sigma_i, sigma_e, _ = cond.calculate_sigma(torso_heart_fibres)

        # -- K_torso:  sigma_i + sigma_e on heart,  sigma_T elsewhere ---
        self.torso_sigma[heart_mask] = sigma_i + sigma_e

        torso_mats = Matrices3D(
            vertices=self.torso_nodes,
            tetrahedrons=self.torso_elems,
            device=self.device, dtype=self.dtype,
        )
        K_torso_coo, _ = torso_mats.assemble_matrices(self.torso_sigma)
        self.K_torso = K_torso_coo.to_sparse_csr()

        # -- K_i:  sigma_i on heart elements, torso node numbering ------
        heart_elems_torso = self.torso_elems[heart_mask]
        heart_mats = Matrices3D(
            vertices=self.torso_nodes,
            tetrahedrons=heart_elems_torso,
            device=self.device, dtype=self.dtype,
        )
        K_i_coo, _ = heart_mats.assemble_matrices(sigma_i)
        self.K_i = K_i_coo.to_sparse_csr()

        # -- heart-to-torso node mapping --------------------------------
        self.heart_to_torso = torch.unique(
            heart_elems_torso.reshape(-1), sorted=True
        )
        self._verify_node_mapping()

        n_heart = self.heart_to_torso.shape[0]
        print(f"  Torso nodes : {self.n_torso:,}")
        print(f"  Heart nodes : {n_heart:,}")
        print(f"  Heart elems : {int(heart_mask.sum()):,}")

    # ------------------------------------------------------------------ #
    def _verify_node_mapping(self):
        """Check the standalone heart mesh matches the torso sub-domain."""
        n_heart  = self.heart_nodes.shape[0]
        n_mapped = self.heart_to_torso.shape[0]
        if n_heart != n_mapped:
            raise RuntimeError(
                f"Node count mismatch: heart mesh {n_heart} vs "
                f"torso subdomain {n_mapped}. "
                f"Ensure heart_tags covers all heart regions."
            )
        torso_sub = self.torso_nodes[self.heart_to_torso]
        diff = (self.heart_nodes - torso_sub).abs().max().item()
        if diff > 1e-3:
            raise RuntimeError(
                f"Node position mismatch (max {diff:.6e}).  "
                f"Heart mesh may not come from this torso mesh."
            )
        print(f"  Node mapping verified (max coord diff = {diff:.2e})")

    # ================================================================== #
    #  Electrodes
    # ================================================================== #
    def load_electrodes(self, filepath, names=None):
        """Load electrode torso-node indices from a .vtx file."""
        if names is None:
            names = ["V1","V2","V3","V4","V5","V6","RA","LA","RL","LL"]
        ids = np.loadtxt(filepath, dtype=np.int64, skiprows=1).tolist()
        if len(ids) != len(names):
            raise ValueError(
                f"Expected {len(names)} electrodes, got {len(ids)}"
            )
        self.electrodes = dict(zip(names, ids))

    # ================================================================== #
    #  Solver preparation  (once, after electrodes are loaded)
    # ================================================================== #
    def _prepare_solver(self):
        """
        Build the float64 system matrix with Neumann BC + penalty ground.

        1. Promote K_torso to float64  (openCARP uses PETSc double).
        2. Add penalty term  alpha * e_g e_g^T  to the ground node.
           This pins Z(g) ~ 0  while keeping the Neumann (no-flux)
           physics on the entire torso surface.
        3. Build Jacobi preconditioner from the penalised matrix.
        4. Promote K_i to float64 for accurate lead-field computation.
        """
        if self._A is not None:
            return                               # already prepared

        ground_idx = self.electrodes[self.ground]

        # K_torso → float64 COO
        K_coo_f64 = self.K_torso.to_sparse_coo().coalesce().to(torch.float64)

        # Penalty parameter: scale from diagonal of K
        idx = K_coo_f64.indices()
        val = K_coo_f64.values()
        diag_vals = val[idx[0] == idx[1]]
        alpha = float(diag_vals.abs().max().item()) * 1e8

        # Penalty matrix  D = alpha * e_g e_g^T
        D_coo = torch.sparse_coo_tensor(
            torch.tensor([[ground_idx], [ground_idx]],
                         device=self.device, dtype=torch.long),
            torch.tensor([alpha], device=self.device, dtype=torch.float64),
            size=K_coo_f64.size(),
            device=self.device, dtype=torch.float64,
        )

        # A = K + D  (Neumann + penalty)
        A_coo = (K_coo_f64 + D_coo).coalesce()
        self._A = A_coo.to_sparse_csr()

        # Jacobi preconditioner (float64)
        self._pcd = Preconditioner()
        self._pcd.create_Jocobi(A_coo)

        # K_i → float64 CSR
        self._K_i_f64 = (
            self.K_i.to_sparse_coo().coalesce().to(torch.float64).to_sparse_csr()
        )

        print(f"  Solver ready  (Neumann + penalty, ground = {self.ground}, "
              f"node {ground_idx}, alpha = {alpha:.2e}, dtype = float64)")

    # ================================================================== #
    #  Reciprocal solve
    # ================================================================== #
    def _solve_reciprocal(self, electrode_name, a_tol, r_tol, max_iter):
        """
        Solve  (K + D) Z  =  delta_e - delta_g   in float64.

        Neumann BC with balanced current: +1 at electrode, -1 at ground.
        The penalty term D pins Z(g) ~ 0  to remove the null space.
        """
        e_idx = self.electrodes[electrode_name]
        g_idx = self.electrodes[self.ground]

        b = torch.zeros(self.n_torso, device=self.device, dtype=torch.float64)
        b[e_idx] =  1.0
        b[g_idx] = -1.0

        cg = ConjugateGradient(self._pcd, self._A, dtype=torch.float64)
        cg.initialize(
            x=torch.zeros(self.n_torso, device=self.device, dtype=torch.float64),
            linear_guess=False,
        )
        Z, n_iter = cg.solve(b, a_tol=a_tol, r_tol=r_tol, max_iter=max_iter)

        if torch.isnan(Z).any():
            raise RuntimeError(f"CG diverged for electrode {electrode_name}")

        # Diagnostics
        Z_heart = Z[self.heart_to_torso]
        print(f"    {electrode_name:>3s}:  CG iters = {n_iter:5d}   "
              f"Z_heart range = [{Z_heart.min():.4e}, {Z_heart.max():.4e}]")
        return Z

    # ================================================================== #
    #  Lead-field precomputation
    # ================================================================== #
    def precompute_electrode(self, name, a_tol, r_tol, max_iter):
        """
        Compute the lead-field vector for one electrode.

            q  =  -(K_i  Z)        ... Eq. (2)

        Only heart-node entries are non-zero; extract them so that
        V(t) = Vm(t) @ q_heart  is a simple  (T, N_heart) @ (N_heart,).
        """
        Z = self._solve_reciprocal(name, a_tol, r_tol, max_iter)

        q_full  = -(self._K_i_f64 @ Z)          # (N_torso,)  float64
        q_heart = q_full[self.heart_to_torso]    # (N_heart,)  float64
        self.q_heart[name] = q_heart.to(self.dtype)

    def precompute_all(self, a_tol=1e-10, r_tol=1e-10, max_iter=20000):
        """Precompute lead fields for every electrode except the ground."""
        self._prepare_solver()
        for name in self.electrodes:
            if name != self.ground:
                self.precompute_electrode(name, a_tol, r_tol, max_iter)

        # Free float64 solver data
        del self._A, self._K_i_f64, self._pcd
        self._A = self._K_i_f64 = self._pcd = None
        torch.cuda.empty_cache()

    # ================================================================== #
    #  ECG computation
    # ================================================================== #
    def unipolar(self, Vm: Tensor, electrode: str) -> Tensor:
        """Unipolar signal  U(t) = Vm(t) @ q_e."""
        return Vm @ self.q_heart[electrode]

    def compute_12lead(self, Vm: Tensor) -> Dict[str, Tensor]:
        """
        Standard 12-lead ECG.

        Limb (Einthoven):   I = LA-RA,  II = LL-RA,  III = LL-LA
        Augmented:          aVR, aVL, aVF
        Precordial:         V1..V6, referenced to Wilson Central Terminal
        """
        Vm = Vm.to(self.device, self.dtype)

        ra = self.unipolar(Vm, "RA")
        la = self.unipolar(Vm, "LA")
        ll = self.unipolar(Vm, "LL")

        I   = la - ra
        II  = ll - ra
        III = ll - la

        aVR = ra - 0.5 * (la + ll)
        aVL = la - 0.5 * (ra + ll)
        aVF = ll - 0.5 * (ra + la)

        wct = (ra + la + ll) / 3.0

        ecg = {
            "I": I, "II": II, "III": III,
            "aVR": aVR, "aVL": aVL, "aVF": aVF,
        }
        for v in ("V1","V2","V3","V4","V5","V6"):
            ecg[v] = self.unipolar(Vm, v) - wct

        # Verify Einthoven's law:  II = I + III  (pointwise)
        err = (II - (I + III)).abs().max().item()
        print(f"  Einthoven check  max|II-(I+III)| = {err:.2e}")

        return ecg

    # ================================================================== #
    #  Plotting
    # ================================================================== #
    def plot_ecg(self, ecg_dict, filename="ecg_12lead.png",
                 dt: Optional[float] = None, smooth_sigma: float = 2.0):
        """
        12-lead ECG in a 3x4 clinical layout.

        smooth_sigma : Gaussian sigma in samples (0 = no smoothing).
        """
        order = [
            "I",   "aVR", "V1", "V4",
            "II",  "aVL", "V2", "V5",
            "III", "aVF", "V3", "V6",
        ]
        fig, axes = plt.subplots(3, 4, figsize=(14, 7), sharex=True)
        for ax, lead in zip(axes.flat, order):
            sig = ecg_dict[lead].detach().cpu().numpy()
            if smooth_sigma > 0:
                sig = gaussian_filter1d(sig, sigma=smooth_sigma)
            if dt is not None:
                t = np.arange(len(sig)) * dt
                ax.plot(t, sig, "k", lw=0.8)
                ax.set_xlabel("ms", fontsize=8)
            else:
                ax.plot(sig, "k", lw=0.8)
            ax.set_title(lead, fontsize=10, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.supylabel("Potential (a.u.)", fontsize=11)
        fig.suptitle("12-Lead ECG", fontsize=13)
        plt.tight_layout(rect=[0.03, 0.0, 1, 0.96])
        plt.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"  Saved {filename}")
