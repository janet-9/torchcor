import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import numpy as np
from assemble import Matrices3DSurface
from preconditioner import Preconditioner
from solver import ConjugateGradient, BiCGStab
from visualize import VTK3DSurface
from reorder import RCM as RCM
import time
from ionic import ModifiedMS2v
from mesh.triangulation import Triangulation
from mesh.materialproperties import MaterialProperties
from mesh.stimulus import Stimulus
from tools import load_stimulus_region, dfmass, sigmaTens
import argparse


parser = argparse.ArgumentParser(description="A simple example of argparse.")
parser.add_argument("-c", '--cuda', type=int, default=0)
parser.add_argument('--vtk', action='store_true')
parser.add_argument('--no-rcm', action='store_false')
args = parser.parse_args()

total_time = time.time()

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
print(device)

T = 5000  # ms = 2.4s
dt = 0.01  # ms
max_iter = 1000
nt = int(T // dt) 

vg = 0.1
diffusl = {1: 0.175, 2: 0.175, 3: 0.175, 4: 0.175}
diffust = {1: 0.175, 2: 0.175, 3: 0.175, 4: 0.175}
tin = {1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15}
tout = {1: 1.5, 2: 1.5, 3: 1.5, 4: 1.5}
topen = {1: 105, 2: 105, 3: 105, 4: 105}
tclose = {1: 185, 2: 185, 3: 185, 4: 185}

apply_rcm = args.no_rcm
print(f"Applying RCM Reordering: {apply_rcm}")

cfgstim1 = {'tstart': 0.0,
            'nstim': 1,
            'period': 100,
            'duration': np.max([0.4, dt]),
            'intensity': 1.0,
            'name': 'crosstim'}

cfgstim2 = {'tstart': 260.0,
            'nstim': 1,
            'period': 100,
            'duration': np.max([0.4, dt]),
            'intensity': 1.0,
            'name': 'crosstim'}


if __name__ == "__main__":
    
    material = MaterialProperties()
    material.add_element_property('sigma_l', 'region', diffusl)
    material.add_element_property('sigma_t', 'region', diffust)

    material.add_nodal_property('tau_in', 'region', tin)
    material.add_nodal_property('tau_out', 'region', tout)
    material.add_nodal_property('tau_open', 'region', topen)
    material.add_nodal_property('tau_close', 'region', tclose)
    material.add_nodal_property('u_gate', 'uniform', vg)
    material.add_nodal_property('u_crit', 'uniform', vg)
    material.add_ud_function('mass', dfmass)
    material.add_ud_function('stiffness', sigmaTens)

    domain = Triangulation()
    domain.readMesh("/home/bzhou6/Projects/FinitePDE/monodomain_equation/trangulated_square_fine_mm.pkl")
    # domain.exportCarpFormat("atrium")
    
    # assign nodal properties
    ionic_model = ModifiedMS2v()
    nodal_properties = material.nodal_property_names()
    point_region_ids = domain.point_region_ids()
    npt = point_region_ids.shape[0]

    for npr in nodal_properties:
        npr_type = material.nodal_property_type(npr)
        attribute_value = ionic_model.get_attribute(npr)

        if attribute_value is not None:
            if npr_type == "uniform":
                values = material.NodalProperty(npr, -1, -1)
            else:
                values = torch.full(size=(npt, 1), fill_value=attribute_value)
                for point_id, region_id in enumerate(point_region_ids):
                    values[point_id] = material.NodalProperty(npr, point_id, region_id)
            ionic_model.set_attribute(npr, values)

    start_time = time.time()

    vertices = torch.from_numpy(domain.Pts()).to(dtype=dtype, device=device)
    triangles = torch.from_numpy(domain.Elems()['Trias'][:, :-1]).to(dtype=torch.long, device=device)

    rcm = RCM(device=device, dtype=dtype)
    if apply_rcm:
        rcm.calculate_rcm_order(vertices, triangles)
        rcm_vertices = rcm.reorder(vertices)
        rcm_triangles = rcm.map(triangles)
    else:
        rcm_vertices = vertices
        rcm_triangles = triangles

    # sigma calculation:
    fibers = torch.from_numpy(domain.Fibres()).to(dtype=dtype, device=device)
    # region_ids = domain.Elems()['Trias'][:, -1]
    sigma_l = diffusl
    sigma_t = diffust
    sigma = sigma_t * torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(fibers.shape[0], 3, 3)
    sigma += (sigma_l - sigma_t) * fibers.unsqueeze(2) @ fibers.unsqueeze(1)

    matrices = Matrices3DSurface(vertices=rcm_vertices, triangles=rcm_triangles, device=device, dtype=dtype)
    K, M = matrices.assemble_matrices(sigma)
    K = K.to(device=device, dtype=dtype)
    M = M.to(device=device, dtype=dtype)
    A = M + K * dt

    assemble_time = time.time() - start_time
    print(f"assemble time: {round(assemble_time, 2)}")

    pcd = Preconditioner()
    pcd.create_Jocobi(A)
    A = A.to_sparse_csr()
    
    Lx = vertices[:, 0].max()
    Ly = vertices[:, 1].max()
    S1 = vertices[:, 0] < 0.05 * Lx
    S2 = np.logical_and(vertices[:, 0] < Lx, vertices[:, 1] < 0.5 * Ly)
    S1 = rcm.reorder(S1)
    S2 = rcm.reorder(S2)

    stimuli = []
    stimulus_1 = Stimulus(cfgstim1)
    stimulus_1.set_stimregion(S1)
    stimuli.append(stimulus_1)
    stimulus_2 = Stimulus(cfgstim2)
    stimulus_2.set_stimregion(S2)
    stimuli.append(stimulus_2)

    # set initial conditions
    u = torch.full(size=(npt,), fill_value=0, device=device, dtype=dtype)
    h = torch.full(size=(npt,), fill_value=1, device=device, dtype=dtype)

    cg = ConjugateGradient(pcd)
    cg.initialize(x=u)

    ts_per_frame = 1000
    ctime = 0
    visualization = VTK3DSurface(vertices.cpu(), triangles.cpu())
    solving_time = time.time()
    for n in range(nt):
        ctime += dt
        I0 = torch.zeros_like(u)
        for stimulus in stimuli:
            I0 += stimulus.stimApp(ctime)
        
        du, dh = ionic_model.differentiate(u, h)
        du = du + I0
        b = u + dt * du
        b = M @ b

        u, total_iter = cg.solve(A, b, a_tol=1e-5, r_tol=1e-5, max_iter=max_iter)
        h += dt * dh

        if total_iter == max_iter:
            print(f"The solution did not converge at {n} iteration")
        else:
            print(f"{n} / {nt}: {total_iter}; {round(time.time() - start_time, 2)}")

        if n % ts_per_frame == 0 and args.vtk:
            visualization.save_frame(color_values=rcm.inverse(u).cpu().numpy() if apply_rcm else u.cpu().numpy(),
                                     frame_path=f"./vtk_files_{len(vertices)}_{apply_rcm}/frame_{n}.vtk")
    print("Solving time: ", time.time() - solving_time)
    print("Total time: ", time.time() - total_time)
