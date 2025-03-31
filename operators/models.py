import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import ModifiedMS2v
from pathlib import Path
from torch_geometric.data import Data
import torch

tc.set_device("cuda:0")
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
simulator.assemble()
mass_matrix, stiffness_matrx = simulator.K.to_sparse_coo(), simulator.M.to_sparse_coo()

node_features = simulator.nodes
edge_index = mass_matrix.indices()  # [2, 2759561]
edge_features = torch.stack([mass_matrix.values(), stiffness_matrx.values()], dim=1)  # [2759561, 2]

mesh_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
