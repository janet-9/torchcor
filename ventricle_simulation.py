from monodomain import VentricleSimulator
from ionic import TenTusscher, MitchellSchaeffer
import torch
from pathlib import Path

simulation_time = 500
dt = 0.05
stim_config_stim_LV_sf = {'tstart': 0.0,
                          'nstim': 1,
                          'period': 800,
                          'duration': max([1.0, dt]),
                          'intensity': 100.0,
                          'name': 'LV_sf'}
stim_config_stim_LV_pf = {'tstart': 0.0,
                          'nstim': 1,
                          'period': 800,
                          'duration': max([1.0, dt]),
                          'intensity': 100.0,
                          'name': 'LV_pf'}
stim_config_stim_LV_af = {'tstart': 0.0,
                          'nstim': 1,
                          'period': 800,
                          'duration': max([1.0, dt]),
                          'intensity': 100.0,
                          'name': 'LV_af'}
stim_config_stim_RV_sf = {'tstart': 5.0,
                          'nstim': 1,
                          'period': 800,
                          'duration': max([1.0, dt]),
                          'intensity': 100.0,
                          'name': 'RV_sf'}
stim_config_stim_RV_mod = {'tstart': 5.0,
                          'nstim': 1,
                          'period': 800,
                          'duration': max([1.0, dt]),
                          'intensity': 100.0,
                          'name': 'RV_mod'}


material_config = {"diffusl": {34: 0.5272, 35: 0.5272, 44: 0.9074, 45: 0.9074, 46: 0.9074},
                   "diffust": {34: 0.2076, 35: 0.2076, 44: 0.3332, 45: 0.3332, 46: 0.3332}}


device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

home_directory = Path.home()

ionic_model = MitchellSchaeffer(device=device)
simulator = VentricleSimulator(ionic_model, T=simulation_time, dt=dt, apply_rcm=True, device=device)
simulator.load_mesh(path=f"{home_directory}/Data/ventricle/biv")
simulator.add_material_property(material_config)

simulator.add_stimulus(f"{home_directory}/Data/ventricle/LV_sf.vtx", stim_config_stim_LV_sf)
simulator.add_stimulus(f"{home_directory}/Data/ventricle/LV_pf.vtx", stim_config_stim_LV_pf)
simulator.add_stimulus(f"{home_directory}/Data/ventricle/LV_af.vtx", stim_config_stim_LV_af)
simulator.add_stimulus(f"{home_directory}/Data/ventricle/RV_sf.vtx", stim_config_stim_RV_sf)
simulator.add_stimulus(f"{home_directory}/Data/ventricle/RV_mod.vtx", stim_config_stim_RV_mod)

simulator.assemble()
simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=1000, plot_interval=10, verbose=True)

