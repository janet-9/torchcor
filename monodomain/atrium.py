from models import Monodomain
from ionic import ModifiedMS2v, CourtemancheRamirezNattel
import torch
from pathlib import Path

simulation_time = 2400
dt = 0.01

device = torch.device(f"cuda:2" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
home_dir = Path.home()

ionic_model = CourtemancheRamirezNattel(dt, device=device, dtype=torch.float32)
# ionic_model.u_gate = 0.1
# ionic_model.u_crit = 0.1
# ionic_model.tau_in = 0.15
# ionic_model.tau_out = 1.5
# ionic_model.tau_open = 105.0
# ionic_model.tau_close = 185.0

simulator = Monodomain(ionic_model, T=simulation_time, dt=dt, device=device, dtype=dtype)
simulator.load_mesh(path=f"{home_dir}/Data/atrium/Case_1")
simulator.add_condutivity(region_ids=[1, 2, 3, 4, 5, 6], il=1.75, it=0.4375)

simulator.add_stimulus(f"{home_dir}/Data/atrium/Case_1/Case_1.vtx", 
                       start=0.0, 
                       duration=2.0, 
                       intensity=1, 
                       period=1, 
                       count=1)

simulator.assemble()
simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=1000, plot_interval=10, verbose=True)