import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import ModifiedMS2v
from pathlib import Path

tc.set_device("cuda:0")
simulation_time = 1500
dt = 0.01

ionic_model = ModifiedMS2v(dt)
ionic_model.u_gate = 0.1
ionic_model.u_crit = 0.1
ionic_model.tau_in = 0.15
ionic_model.tau_out = 1.5
ionic_model.tau_open = 105.0
ionic_model.tau_close = 185.0

home_dir = Path("./surface")
n_nodes_list = sorted([int(sub_dir.name) for sub_dir in home_dir.iterdir()])
print(n_nodes_list)

for n_nodes in n_nodes_list:
    mesh_dir = f"{home_dir}/{n_nodes}/"

    simulator = Monodomain(ionic_model, T=simulation_time, dt=dt)
    simulator.load_mesh(path=mesh_dir, unit_conversion=1000)
    simulator.add_condutivity(region_ids=[0], il=0.4, it=0.4)

    simulator.add_stimulus(f"{mesh_dir}/{0}.vtx", 
                        start=0.0, 
                        duration=2.0, 
                        intensity=50, 
                        period=500, 
                        count=3)

    simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=100, plot_interval=10, verbose=True, format="")



# ModifiedMS2v 80472 90.04 29.31 59.28 5110 58.99 7.29
# ModifiedMS2v 205341 114.11 29.41 83.29 12022 73.15 7.58
# ModifiedMS2v 319962 146.87 29.5 115.95 16128 81.16 7.95
# ModifiedMS2v 569404 228.38 30.12 196.8 24409 93.0 8.7
# ModifiedMS2v 1279134 514.36 71.74 441.16 48074 97.94 10.36
# ModifiedMS2v 5105726 2357.02 285.74 2069.8 138076 99.0 20.3
# ModifiedMS2v 6313331 3003.2 350.76 2650.92 161125 99.79 27.58