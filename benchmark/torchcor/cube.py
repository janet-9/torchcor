import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import TenTusscherPanfilov
from pathlib import Path

tc.set_device("cuda:0")
simulation_time = 500
dt = 0.005

home_dir = Path("./volume")

n_nodes_list = sorted([int(sub_dir.name) for sub_dir in home_dir.iterdir()])
print(n_nodes_list)

for n_nodes in n_nodes_list:
    ionic_model = TenTusscherPanfilov(cell_type="ENDO", dt=dt)
    simulator = Monodomain(ionic_model, T=simulation_time, dt=dt)
    simulator.load_mesh(path=f"{home_dir}/{n_nodes}", unit_conversion=1000)
    simulator.add_condutivity([0], il=0.17, it=0.019, el=0.62, et=0.2)

    simulator.add_stimulus(f"{home_dir}/{n_nodes}/{n_nodes}.vtx", start=0.0, duration=2.0, intensity=50)

    simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=100, plot_interval=10, verbose=True, format="")


# TenTusscherPanfilov 14706 206.52 5042 38.59 7.26
# TenTusscherPanfilov 44831 207.85 4066 46.24 7.55
# TenTusscherPanfilov 325186 325.17 1058 97.0 10.69
# TenTusscherPanfilov 444870 434.72 479 98.0 13.01
# TenTusscherPanfilov 620739 603.74 181 99.0 15.81
# TenTusscherPanfilov 920028 941.69 53 100.0 20.0
# TenTusscherPanfilov 1450392 1683.22 2 100.0 26.65