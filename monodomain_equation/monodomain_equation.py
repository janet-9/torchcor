import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from collections import deque
import numpy as np
from assemble import Matrices3DSurface
from preconditioner import Preconditioner
from solver import ConjugateGradient
from visualize import VTK3DSurface
from boundary import apply_dirichlet_boundary_conditions
import time
from ionic import ModifiedMS2v
from mesh.triangulation import Triangulation
from mesh.materialproperties import MaterialProperties
from mesh.stimulus import Stimulus
from tools import load_stimulus_region, dfmass, sigmaTens
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
print(device)


dt = 0.01
vg = 0.1
diffusl = (1000 * 1000) * 0.175
diffust = (1000 * 1000) * 0.04375
tin = 0.15
tout = 1.5
topen = 105.0
tclose = 185.0

use_renumbering = True
T = 2400


cfgstim1 = {'tstart': 0.0,
            'nstim': 3,
            'period': 800,
            'duration': np.max([2.0, dt]),
            'intensity': 1.0,
            'name': 'S1'}


if __name__ == "__main__":
    
    material = MaterialProperties()
    material.add_element_property('sigma_l', 'uniform', diffusl)
    material.add_element_property('sigma_t', 'uniform', diffust)

    material.add_nodal_property('tau_in', 'uniform', tin)
    material.add_nodal_property('tau_out', 'uniform', tout)
    material.add_nodal_property('tau_open', 'uniform', topen)
    material.add_nodal_property('tau_close', 'uniform', tclose)
    material.add_nodal_property('u_gate', 'uniform', vg)
    material.add_nodal_property('u_crit', 'uniform', vg)
    material.add_ud_function('mass', dfmass)
    material.add_ud_function('stiffness', sigmaTens)

    domain = Triangulation()
    domain.readMesh("/home/bzhou6/Projects/FinitePDE/data/Case_1")
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
    # sigma calculation:
    fibers = torch.from_numpy(domain.Fibres())
    # region_ids = domain.Elems()['Trias'][:, -1]
    sigma_l = diffusl
    sigma_t = diffust
    sigma = sigma_t * torch.eye(3).unsqueeze(0).expand(fibers.shape[0], 3, 3)
    sigma += (sigma_l - sigma_t) * fibers.unsqueeze(2) @ fibers.unsqueeze(1)
    sigma = sigma.to(dtype=dtype, device=device)
    # assemble matrix
    vertices = torch.from_numpy(domain.Pts())
    triangles = torch.from_numpy(domain.Elems()['Trias'][:, :-1])
    matrices = Matrices3DSurface(vertices=vertices, triangles=triangles, device=device, dtype=dtype)
    K, M = matrices.assemble_matrices(sigma)
    K = K.to(device=device, dtype=dtype)
    M = M.to(device=device, dtype=dtype)
    A = M + K * dt
    assemble_time = time.time() - start_time
    print(f"assemble time: {round(assemble_time, 2)}")

    pcd = Preconditioner()
    pcd.create_Jocobi(A)

    pointlist = load_stimulus_region('/home/bzhou6/Projects/FinitePDE/data/Case_1.vtx')  # (2168,)
    S1 = torch.zeros(size=(npt,), device=device, dtype=torch.bool)
    S1[pointlist] = True
    stimulus = Stimulus(cfgstim1)
    stimulus.set_stimregion(S1)
    stimuli = [stimulus]

    # set initial conditions
    u = torch.full(size=(npt,), fill_value=0, device=device, dtype=dtype)
    h = torch.full(size=(npt,), fill_value=1, device=device, dtype=dtype)

    cg = ConjugateGradient(pcd)
    cg.initialize(x=u)
    
    start_time = time.time()
    stable_list = deque(maxlen=10)
    max_iter = 1000
    nt = int(T // dt)
    ts_per_frame = 1000
    ctime = 0
    frames = [(0, u)]
    visualization = VTK3DSurface(vertices, triangles)
    for n in range(nt):
        ctime += dt
        du, dh = ionic_model.differentiate(u, h)
        b = u + dt * du
        for stimulus in stimuli:
            I0 = stimulus.stimApp(ctime)
            b = b + dt * I0
        b = M @ b

        u, total_iter = cg.solve(A, b, a_tol=1e-5, r_tol=1e-5, max_iter=max_iter)
        h = h + dt * dh

        stable_list.append(total_iter)
        if sum(stable_list) == stable_list.maxlen:
            break

        if total_iter == max_iter:
            print(f"The solution did not converge at {n} iteration")
        else:
            print(f"{n} / {nt}: {total_iter}; {round(time.time() - start_time, 2)}")

        # if n % ts_per_frame == 0:
        #     # frames.append((n, u))

        #     visualization.save_frame(color_values=u.cpu().numpy(),
        #                              frame_path=f"./vtk_files_{len(vertices)}/frame_{n}.vtk")


