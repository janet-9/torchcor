from monodomain import AtriumSimulatorCourtemanche
from ionic import CourtemancheRamirezNattel
import torch
from pathlib import Path


simulation_time = 2400
dt = 0.01
stim_config = {'tstart': 0.0,
               'nstim': 3,
               'period': 800,
               'duration': max([2.0, dt]),
               'intensity': 100.0,
               'name': 'S1'}
material_config = {"vg": 0.1,
                   "diffusl": 0.4 * 1000 * 1000,
                   "diffust": 0.4 * 1000 * 1000,
                   "tin": 0.15,
                   "tout": 1.5,
                   "topen": 105.0,
                   "tclose": 185.0}

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
home_directory = Path.home()

ionic_model = CourtemancheRamirezNattel(dt=dt, device=device)
simulator = AtriumSimulatorCourtemanche(ionic_model, T=simulation_time, dt=dt, apply_rcm=True, device=device)
simulator.load_mesh(path=f"{home_directory}/Data/atrium/Case_1")
simulator.add_material_property(material_config)
simulator.set_stimulus_region(path=f"{home_directory}/Data/atrium/Case_1.vtx")
simulator.add_stimulus(stim_config)
simulator.assemble()
simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=1000, plot_interval=simulation_time, verbose=False)

