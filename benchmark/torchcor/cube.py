import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import TenTusscherPanfilov
from pathlib import Path

tc.set_device("cuda:0")
simulation_time = 20
dt = 0.005

home_dir = "./volume"
n_nodes = 3703

ionic_model = TenTusscherPanfilov(cell_type="ENDO", dt=dt)
simulator = Monodomain(ionic_model, T=simulation_time, dt=dt)
simulator.load_mesh(path=f"{home_dir}/{n_nodes}", unit_conversion=1000)
simulator.add_condutivity([0], il=0.17, it=0.019, el=0.62, et=0.2)

simulator.add_stimulus(f"{home_dir}/{n_nodes}/{n_nodes}.vtx", start=0.0, duration=2.0, intensity=50)

simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=100, plot_interval=1, verbose=True, format="vtk")

