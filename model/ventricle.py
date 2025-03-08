from monodomain import Monodomain
from ionic import TenTusscherPanfilov
import torch
from pathlib import Path

simulation_time = 500
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

ionic_model = TenTusscherPanfilov(cell_type="ENDO", dt=dt, device=device, dtype=torch.float32)
simulator = Monodomain(ionic_model, T=simulation_time, dt=dt, device=device, dtype=torch.float32)
simulator.load_mesh(path=f"{home_dir}/Data/ventricle/")
simulator.add_condutivity([34, 35], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
simulator.add_condutivity([44, 45, 46], il=0.9074, it=0.3332, el=0.9074, et=0.3332)

simulator.add_stimulus(f"{home_dir}/Data/ventricle/LV_sf.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(f"{home_dir}/Data/ventricle/LV_pf.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(f"{home_dir}/Data/ventricle/LV_af.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(f"{home_dir}/Data/ventricle/RV_sf.vtx", start=5.0, duration=1.0, intensity=100)
simulator.add_stimulus(f"{home_dir}/Data/ventricle/RV_mod.vtx", start=5.0, duration=1.0, intensity=100)

simulator.assemble()
simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=100, plot_interval=10, verbose=True)

