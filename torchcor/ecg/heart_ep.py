import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import TenTusscherPanfilov
from pathlib import Path

tc.set_device("cuda:1")
dtype = tc.float64
simulation_time = 500
dt = 0.01

mesh_dir = Path("/data/Bei/Torso/HC2/heart")
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

snapshot_interval = 1
Vm = simulator.solve(a_tol=1e-5,              # absolute tolerance
                     r_tol=1e-5,              # relative tolerance
                     max_iter=100,            # maximum number of iterations for each CG calculation
                     snapshot_interval=snapshot_interval,     # save the soluation after every 1 ms
                     verbose=True,
                     result_path="./biventricle")  # the folder in which the results are saved

# POSTPROCESSING: 
ATs = simulator.compute_activation_map(Vm=Vm, 
                                       snapshot_interval=snapshot_interval, 
                                       threshold=0,
                                       save=False)
print("ATs: ", ATs.min().item(), ATs.cpu().max().item(), flush=True)
RTs = simulator.compute_repolarization_map(Vm=Vm, 
                                           search_after=ATs,
                                           snapshot_interval=snapshot_interval, 
                                           threshold=-70,
                                           save=False)
print("RTs: ", RTs.min().item(), RTs.cpu().max().item(), flush=True)
