"""
Full pipeline: cardiac EP simulation -> ECG forward problem -> 12-lead ECG plot.
"""
import sys
import os
import torch
from pathlib import Path

# Ensure torchcor is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parent)

import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import TenTusscherPanfilov
from ecg.leadfield import LeadField

# ---------- Config ----------
device = torch.device("cuda:0")
dtype = torch.float64
tc.set_device("cuda:0")

mesh_dir = Path("/data/Bei/Torso/HC2/heart")
torso_mesh_dir = "/data/Bei/Torso/HC2/mesh"
heart_mesh_dir = "/data/Bei/Torso/HC2/heart"
electrode_file = "/data/Bei/Torso/HC2/electrodes/lf_src.vtx"

vm_file = Path("Vm_saved.pt")

# ========== STEP 1: Cardiac EP simulation (skip if Vm already exists) ==========
if vm_file.exists():
    print(f"Loading saved Vm from {vm_file}")
    Vm = torch.load(vm_file, weights_only=True)
    print(f"Vm shape: {Vm.shape}")
else:
    print("=" * 60)
    print("STEP 1: Running cardiac EP simulation (500 ms)")
    print("=" * 60)

    simulation_time = 500
    dt = 0.01
    snapshot_interval = 1

    im = TenTusscherPanfilov(cell_type="ENDO", dt=dt, dtype=dtype)
    simulator = Monodomain(ionic_models=[im], T=simulation_time, dt=dt, dtype=dtype)
    simulator.load_mesh(path=mesh_dir)
    simulator.add_conductivity([24, 25], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
    simulator.add_conductivity([34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332)

    simulator.add_stimulus(mesh_dir / "pacing" / "LV_sf.vtx", start=0.0, duration=1.0, intensity=100)
    simulator.add_stimulus(mesh_dir / "pacing" / "LV_pf.vtx", start=0.0, duration=1.0, intensity=100)
    simulator.add_stimulus(mesh_dir / "pacing" / "LV_af.vtx", start=0.0, duration=1.0, intensity=100)
    simulator.add_stimulus(mesh_dir / "pacing" / "RV_sf.vtx", start=5.0, duration=1.0, intensity=100)
    simulator.add_stimulus(mesh_dir / "pacing" / "RV_mod.vtx", start=5.0, duration=1.0, intensity=100)

    Vm = simulator.solve(
        a_tol=1e-5, r_tol=1e-5, max_iter=100,
        snapshot_interval=snapshot_interval,
        verbose=True,
        result_path="./biventricle",
    )
    print(f"Vm shape: {Vm.shape}")
    Vm = Vm.cpu()
    torch.save(Vm, vm_file)
    print(f"Saved Vm to {vm_file}")
    del simulator, im
    torch.cuda.empty_cache()

# ========== STEP 2: Build lead field solver ==========
print("\n" + "=" * 60)
print("STEP 2: Assembling ECG lead field matrices")
print("=" * 60)

lf = LeadField(torso_mesh_dir, heart_mesh_dir, device=device, dtype=dtype)

# Torso conductivities (S/m, isotropic)
lf.add_torso_conductivity([20, 21], g=0.25)
lf.add_torso_conductivity([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23], g=0.6667)
lf.add_torso_conductivity([3], g=0.05)
lf.add_torso_conductivity([1, 7], g=0.2472)
lf.add_torso_conductivity([4], g=0.1667)
lf.add_torso_conductivity([5], g=0.1667)
lf.add_torso_conductivity([6], g=0.0714)
lf.add_torso_conductivity([2], g=0.117)
lf.add_torso_conductivity([8], g=0.1)
lf.add_torso_conductivity([9], g=0.1)

# Heart conductivities (S/m, anisotropic)
lf.add_heart_conductivity([24, 25], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
lf.add_heart_conductivity([34, 35, 36], il=0.9074, it=0.3332, el=0.9074, et=0.3332)

lf.build()

# ========== STEP 3: Load electrodes & precompute lead fields ==========
print("\n" + "=" * 60)
print("STEP 3: Precomputing lead fields (9 CG solves)")
print("=" * 60)

# Electrode order in lf_src.vtx (confirmed with dataset owner):
#   V1, V2, V3, V4, V5, V6, RA, LA, RL, LL
# RL is the reference/ground.
lf.load_electrodes(electrode_file)   # uses the default names above
lf.precompute_all(a_tol=1e-8, r_tol=1e-8, max_iter=20000)

# ========== STEP 4: Compute & plot 12-lead ECG ==========
print("\n" + "=" * 60)
print("STEP 4: Computing 12-lead ECG")
print("=" * 60)

# Keep Vm on CPU: the final Vm @ q_heart product is cheap (one per
# electrode) and avoids shipping a multi-GB tensor to the GPU.
ecg12 = lf.compute_12lead(Vm.cpu())
lf.plot_ecg(ecg12, filename="ecg_12lead.png", dt=1.0)
print("Done.")
