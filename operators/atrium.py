import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import ModifiedMS2v
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="get the result",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-case_id", type=int, default=1)
args = parser.parse_args()

tc.set_device("cuda:1")
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

for i in range(1, 101):

    # case_name = f"Case_{args.case_id}"
    case_name = f"Case_{i}"
    mesh_dir = Path("/data/Bei/meshes_refined") / case_name
    for filepath in mesh_dir.iterdir():
        if filepath.suffix == ".vtx":
            with filepath.open("r") as f:
                num_pacing_points = int(f.readline().strip())
                if num_pacing_points == 0:
                    continue
            
            simulator = Monodomain(ionic_model, T=simulation_time, dt=dt, dtype=dtype)
            simulator.load_mesh(path=mesh_dir, unit_conversion=1000)
            simulator.add_condutivity(region_ids=[1, 2, 3, 4, 5, 6], il=0.4, it=0.4)

            simulator.add_stimulus(filepath, 
                                   start=0.0, 
                                   duration=2.0, 
                                   intensity=50)

            simulator.solve(a_tol=1e-5, 
                            r_tol=1e-5, 
                            max_iter=100, 
                            calculate_AT_RT=True,
                            linear_guess=True,
                            snapshot_interval=simulation_time, 
                            verbose=True,
                            result_path=Path("/data/Bei/atrium_at_rt") / filepath.stem)
