from monodomain import Monodomain
from ionic import ModifiedMS2v, CourtemancheRamirezNattel
import torch
from pathlib import Path

simulation_time = 1500
dt = 0.01

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(device_id)
    gpu_properties = torch.cuda.get_device_properties(device_id)
    total_memory = gpu_properties.total_memory / (1024 ** 3)  # Convert bytes to GB

    print(f"GPU: {gpu_name}")
    print(f"Total Memory: {total_memory:.2f} GB")
else:
    print("No GPU available.")

home_dir = Path.home()

# ionic_model = ModifiedMS2v(dt, device=device, dtype=torch.float32)
# ionic_model.u_gate = 0.1
# ionic_model.u_crit = 0.1
# ionic_model.tau_in = 0.15
# ionic_model.tau_out = 1.5
# ionic_model.tau_open = 105.0
# ionic_model.tau_close = 185.0

ionic_model = CourtemancheRamirezNattel(dt, device=device, dtype=torch.float32)

# for i in range(1, 101):
#     case_name = f"Case_{i}"
#     print(case_name, end=" ")
#     mesh_dir = f"{home_dir}/Data/left_atrium_100/{case_name}"

#     simulator = Monodomain(ionic_model, T=simulation_time, dt=dt, device=device, dtype=torch.float32)
#     simulator.load_mesh(path=mesh_dir, unit_conversion=1000)
#     simulator.add_condutivity(region_ids=[1, 2, 3, 4, 5, 6], il=1.75, it=0.4375)

#     simulator.add_stimulus(f"{mesh_dir}/{case_name}.vtx", 
#                         start=0.0, 
#                         duration=2.0, 
#                         intensity=50, 
#                         period=800, 
#                         count=3)

#     simulator.assemble()
#     simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=100, plot_interval=10, verbose=True)

case_name = f"Case_1"
print(case_name, end=" ")
mesh_dir = f"/home/bzhou6/Data/atrium/{case_name}/"

simulator = Monodomain(ionic_model, T=simulation_time, dt=dt, device=device, dtype=torch.float32)
simulator.load_mesh(path=mesh_dir, unit_conversion=1000)
simulator.add_condutivity(region_ids=[1, 2, 3, 4, 5, 6], il=0.4, it=0.4, el=0.4, et=0.4)

simulator.add_stimulus(f"{mesh_dir}/{case_name}.vtx", 
                       start=0.0, 
                       duration=2.0, 
                       intensity=50, 
                       period=500, 
                       count=3)

simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=100, plot_interval=10, verbose=True)