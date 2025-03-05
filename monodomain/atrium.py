from models import Monodomain
from ionic import ModifiedMS2v
import torch
from pathlib import Path

simulation_time = 2400
dt = 0.01

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
home_dir = Path.home()

ionic_model = ModifiedMS2v(device=device, dtype=dtype)
ionic_model.u_gate = 0.1
ionic_model.u_crit = 0.1
ionic_model.tau_in = 0.15
ionic_model.tau_out = 1.5
ionic_model.tau_open = 105.0
ionic_model.tau_close = 185.0


simulator = Monodomain(ionic_model, T=simulation_time, dt=dt, device=device, dtype=dtype)
simulator.load_mesh(path=f"{home_dir}/Data/ventricle/")
simulator.add_condutivity([1, 2, 3, 4, 5, 6], il=0.5272, it=0.2076, el=1.0732, et=0.4227)

simulator.add_stimulus(f"{home_dir}/Data/ventricle/LV_sf.vtx", 
                       name="S1", 
                       start=0.0, 
                       duration=2.0, 
                       intensity=1, 
                       period=800, 
                       times=3)

simulator.assemble()
simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=1000, plot_interval=10, verbose=True)