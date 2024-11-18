from monodomain import VentricleSimulator
from ionic import TenTusscher
import torch
from pathlib import Path

simulation_time = 500
dt = 0.05
stim_config = {'tstart': 0.0,  'nstim': 1,   'period': 800,   'duration': max([1.0, dt]),  'intensity': 100.0,   'name': 'LV_sf'}
# stim_config2 = {'tstart': 0.0,  'nstim': 1,   'period': 800,   'duration': max([1.0, dt]),  'intensity': 100.0,   'name': 'S1'}
#stim_config2 = {'tstart': 0.0,  'nstim': 3,   'period': 800,   'duration': max([2.0, dt]),  'intensity': 100.0,   'name': 'S1'}


material_config = { "diffusl": (1000 * 1000) * 0.175,
                   "diffust": (1000 * 1000) * 0.04375}


device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

home_directory = Path.home()

ionic_model = TenTusscher(device=device)
simulator = VentricleSimulator(ionic_model, T=simulation_time, dt=dt, apply_rcm=True, device=device)
simulator.load_mesh(path=f"{home_directory}/Data/ventricle/biv")
simulator.add_material_property(material_config)
simulator.set_stimulus_region(f"{home_directory}/Data/ventricle/LV_sf.vtx")
simulator.add_stimulus(stim_config)


simulator.assemble()
simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=1000, plot_interval=10, verbose=True)

