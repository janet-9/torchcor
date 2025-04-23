import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import ModifiedMS2v
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Case id",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-case_id", type=int, default=2)
args = parser.parse_args()

tc.set_device("cuda:0")
dtype = tc.float32
simulation_time = 600
dt = 0.01

ionic_model = ModifiedMS2v(dt, dtype=dtype)
ionic_model.u_gate = 0.1
ionic_model.u_crit = 0.1
ionic_model.tau_in = 0.15
ionic_model.tau_out = 1.5
ionic_model.tau_open = 105.0
ionic_model.tau_close = 185.0

data_dir = Path("/data/scratch/acw554")
for il in range(1, 21):
    for it in range(1, 21):
        case_name = f"Case_{args.case_id}"
        print(case_name, il/10, it/10, flush=True)
        mesh_dir = data_dir / "meshes_refined" / case_name
        vtk_filepath = mesh_dir / f"{case_name}.vtx"
                
        simulator = Monodomain(ionic_model, T=simulation_time, dt=dt, dtype=dtype)
        simulator.load_mesh(path=mesh_dir, unit_conversion=1000)
        simulator.add_condutivity(region_ids=[1, 2, 3, 4, 5, 6], il=il/10, it=it/10)

        simulator.add_stimulus(vtk_filepath, 
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
                        result_path=data_dir / "atrium_conductivity_600" / case_name / f"{il/10}_{it/10}")
