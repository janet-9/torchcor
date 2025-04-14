import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import ModifiedMS2v, CourtemancheRamirezNattel
from pathlib import Path

tc.set_device("cuda:1")
dtype = tc.float32
simulation_time = 500
dt = 0.01

ionic_model = ModifiedMS2v(dt, dtype=dtype)
ionic_model.u_gate = 0.1
ionic_model.u_crit = 0.1
ionic_model.tau_in = 0.15
ionic_model.tau_out = 1.5
ionic_model.tau_open = 105.0
ionic_model.tau_close = 185.0

# ionic_model = CourtemancheRamirezNattel(dt)

case_name = f"Case_1"
mesh_dir = f"{Path.home()}/Data/atrium/{case_name}/"

simulator = Monodomain(ionic_model, T=simulation_time, dt=dt, dtype=dtype)
simulator.load_mesh(path=mesh_dir, unit_conversion=1000)
simulator.add_condutivity(region_ids=[1, 2, 3, 4, 5, 6], il=0.4, it=0.4)

simulator.add_stimulus(f"{mesh_dir}/{case_name}.vtx", 
                       start=0.0, 
                       duration=2.0, 
                       intensity=50)

simulator.solve(a_tol=1e-5, 
                r_tol=1e-5, 
                max_iter=100, 
                calculate_AT_RT=True,
                linear_guess=False,
                snapshot_interval=1, 
                verbose=True,
                result_folder=str(dtype)[-2:])

# simulator.pt_to_vtk()
# simulator.phie_recovery()