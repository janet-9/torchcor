import numpy as np
import torch
from monodomain.simulator import AtrialSimulator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

simulation_time = 2400
dt = 0.01

stim_config = {'tstart': 0.0,
               'nstim': 3,
               'period': 800,
               'duration': np.max([2.0, dt]),
               'intensity': 1.0,
               'name': 'S1'}

material_config = {"vg": 0.1,
                   "diffusl": (1000 * 1000) * 0.175,
                   "diffust": (1000 * 1000) * 0.04375,
                   "tin": 0.15,
                   "tout": 1.5,
                   "topen": 105.0,
                   "tclose": 185.0}

simulator = AtrialSimulator(T=simulation_time, dt=dt, apply_rcm=True,  device=device, dtype=dtype)
simulator.load_mesh(path="/Users/bei/Project/FinitePDE/data/Case_1")
simulator.add_material_property(material_config)
simulator.assemble()
simulator.set_stimulus_region(path="/Users/bei/Project/FinitePDE/data/Case_1.vtx")
simulator.add_stimulus(stim_config)
simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=1000)