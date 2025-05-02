from scipy.stats import qmc
import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import ModifiedMS2v
from pathlib import Path
import argparse
import torch
import time
import numpy as np
import shutil


class Domain(Monodomain):
    def __init__(self, ionic_model, T, dt, device=tc.get_device(), dtype=None):
        super().__init__(ionic_model, T, dt, device, dtype)

    def solve(self, a_tol, r_tol, max_iter, calculate_AT_RT=True, linear_guess=True, snapshot_interval=1, verbose=True, result_path=None):
        self.result_path = Path(result_path)
        self.result_path.mkdir(parents=True, exist_ok=True)

        self.assemble()
        
        u = self.ionic_model.initialize(self.n_nodes)
        u_initial = u.clone()
        self.cg.initialize(x=u, linear_guess=linear_guess)
        ts_per_frame = int(snapshot_interval / self.dt)
    
        if calculate_AT_RT:
            activation_time = torch.ones_like(u_initial) * -1
            repolarization_time = torch.ones_like(u_initial) * -1

        t = 0
        solving_time = time.time()
        total_ionic_time = 0
        total_electric_time = 0
        n_total_iter = 0
        solution_list = [u_initial]
        for n in range(1, self.nt + 1):
            t += self.dt
            
            ### CG step ###
            u, n_iter, ionic_time, electric_time = self.step(u, t, a_tol, r_tol, max_iter, verbose)
            n_total_iter += n_iter
            total_ionic_time += ionic_time
            total_electric_time += electric_time
            
            ### calculate AT and RT ###
            if calculate_AT_RT:
                activation_time[(u > 0) & (activation_time == -1)] = t
                repolarization_time[(activation_time > 0) & (repolarization_time == -1) & (u < -70)] = t

            ### keep track of GPU usage ###
            if n % ts_per_frame == 0:
                solution_list.append(u.clone())
                
                if verbose and snapshot_interval != self.T:
                    print(f"t: {round(t, 1)}/{self.T}", 
                          f"Time elapsed:", round(time.time() - solving_time, 2),
                          f"Total CG iter:", n_total_iter,
                          flush=True)

        ### save AT, RT, and solutions to disk ###
        if calculate_AT_RT:
            torch.save(activation_time.cpu(), self.result_path / "ATs.pt")
            torch.save(repolarization_time.cpu(), self.result_path / "RTs.pt")

        if snapshot_interval < self.T:
            torch.save(torch.stack(solution_list, dim=0).cpu(), self.result_path / "Vm.pt")

        ### print log info to console ###
        if verbose:
            print(self.ionic_model.name,
                  self.n_nodes, 
                  round(time.time() - solving_time, 2),
                  round(total_ionic_time, 2),
                  round(total_electric_time, 2),
                  n_total_iter,
                  flush=True)
            
            if calculate_AT_RT:
                print("ATs: ", activation_time.cpu().min().item(), activation_time.cpu().max().item(), flush=True)
                print("RTs: ", repolarization_time.cpu().min().item(), repolarization_time.cpu().max().item(), flush=True)

        return n_total_iter



parser = argparse.ArgumentParser(description="Case id",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-case_id", type=int, default=2)
args = parser.parse_args()

tc.set_device("cuda:1")
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


# sampling from latin hypercube
num_normal_regions   = 3
num_fibrotic_regions = 3
num_regions          = num_normal_regions + num_fibrotic_regions
num_dim              = num_regions * 2
num_samples          = 300 # num_dim*10

sampler = qmc.LatinHypercube(d=num_dim)
lb      = [0.1] * num_normal_regions + [0.1] * num_fibrotic_regions + [1.0] * num_normal_regions+ [1.0] * num_fibrotic_regions
ub      = [1.0] * num_normal_regions + [0.5] * num_fibrotic_regions + [5.0] * num_normal_regions+ [8.0] * num_fibrotic_regions
samples = sampler.random(num_samples)
samples = qmc.scale(samples, lb, ub)

g_it = samples[:, :6]
g_il = samples[:, :6] * samples[:, 6:]


case_name = f"Case_{args.case_id}"
data_dir = Path("/data/scratch/acw554")
# data_dir = Path("/data/Bei")
output_folder_name = "latine_conductivity"
for sample_id, (il, it) in enumerate(zip(g_il, g_it)):
    print(case_name, il.max(), it.max(), flush=True)
    mesh_dir = data_dir / "meshes_refined" / case_name
    vtk_filepath = mesh_dir / f"{case_name}.vtx"
    
    simulator = Domain(ionic_model, T=simulation_time, dt=dt, dtype=dtype)
    simulator.load_mesh(path=mesh_dir, unit_conversion=1000)

    for region_id, (l, t) in enumerate(zip(il, it)):
        simulator.add_condutivity(region_ids=[region_id + 1], il=l, it=t)

    simulator.add_stimulus(vtk_filepath, 
                           start=0.0, 
                           duration=2.0, 
                           intensity=50)
    
    result_path = data_dir / output_folder_name / case_name / f"{sample_id}"
    simulator.solve(a_tol=1e-5, 
                    r_tol=1e-5, 
                    max_iter=100, 
                    calculate_AT_RT=True,
                    linear_guess=True,
                    snapshot_interval=simulation_time, 
                    verbose=True,
                    result_path=result_path)
    
    np.savez(result_path / "conductivity.npz", g_il=il, g_it=it)