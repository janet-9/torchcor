from monodomain.simulator import AtrialSimulator
from ionic import ModifiedMS2v, TenTusscher
import torch

simulation_time = 2400
dt = 0.01
stim_config = {'tstart': 0.0,
               'nstim': 3,
               'period': 800,
               'duration': max([2.0, dt]),
               'intensity': 1.0,
               'name': 'S1'}
material_config = {"vg": 0.1,
                   "diffusl": (1000 * 1000) * 0.175,
                   "diffust": (1000 * 1000) * 0.04375,
                   "tin": 0.15,
                   "tout": 1.5,
                   "topen": 105.0,
                   "tclose": 185.0}

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

ionic_model = TenTusscher(device=device)
simulator = AtrialSimulator(ionic_model, T=simulation_time, dt=dt, apply_rcm=True, device=device)
simulator.load_mesh(path="./data/atrium/Case_1")
simulator.add_material_property(material_config)
simulator.set_stimulus_region(path="./data/atrium/Case_1.vtx")
simulator.add_stimulus(stim_config)
simulator.assemble()
simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=1000, plot_interval=10, verbose=True)

