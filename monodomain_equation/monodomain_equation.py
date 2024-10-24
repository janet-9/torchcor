import torch
import numpy as np
from assemble import Matrices
from preconditioner import Preconditioner
from sovler import ConjugateGradient
from utils import Visualization
from matplotlib import tri
from boundary import apply_dirichlet_boundary_conditions
import time
from ionic import ModifiedMS2v
from material import Properties
import os
from mesh.triangulation import Triangulation
from mesh.materialproperties import MaterialProperties
from mesh.stimulus import Stimulus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
print(device)

def load_stimulus_region(vtxfile: str) -> np.ndarray:
    """ load_stimulus_region(vtxfile) reads the file vtxfile to
    extract point IDs where stimulus will be applied
    """
    with open(vtxfile,'r') as fstim:
        nodes = fstim.read()
        nodes = nodes.strip().split()
    npt       = int(nodes[0])
    nodes     = nodes[2:]
    pointlist = -1.0*np.ones(shape=npt,dtype=int)
    for jj,inod in enumerate(nodes):
        pointlist[jj] = int(inod)
    return(pointlist.astype(int))


def dfmass(elemtype:str, iElem:int,domain:Triangulation,matprop:MaterialProperties):
    """ empty function for mass properties"""
    return(None)

def sigmaTens(elemtype:str, iElem:int,domain:Triangulation,matprop:MaterialProperties) -> np.ndarray :
    """ function to evaluate the diffusion tensor """
    fib   = domain.Fibres()[iElem,:]
    rID   = domain.Elems()[elemtype][iElem,-1]
    sigma_l = matprop.ElementProperty('sigma_l',elemtype,iElem,rID)
    sigma_t = matprop.ElementProperty('sigma_t',elemtype,iElem,rID)
    Sigma = sigma_t *np.eye(3)
    for ii in range(3):
        for jj in range(3):
            Sigma[ii,jj] = Sigma[ii,jj]+ (sigma_l-sigma_t)*fib[ii]*fib[jj]
    return(Sigma)




#
# start = time.time()
# print("assembling matrices")
# matrices = Matrices(device=device, dtype=dtype)
# K, M = matrices.assemble_matrices(triangulation, alpha)
# K = K.to(device=device, dtype=dtype)
# M = M.to(device=device, dtype=dtype)
# print(f"assembled in: {time.time() - start} seconds")
#
# # print(K.to_dense().numpy())
#
#
#
#
#
# # apply initial condition for A
# print("applying boundary condition for A")
# dirichlet_boundary_nodes = torch.arange(20 * Nx, 20 * Nx + Ny, device=device)
# boundary_values = torch.ones_like(dirichlet_boundary_nodes, device=device, dtype=dtype) * T0
#
# A = M + dt * K
# A = apply_dirichlet_boundary_conditions(A, dirichlet_boundary_nodes)
#
# pcd = Preconditioner()
# pcd.create_Jocobi(A)
# cg = ConjugateGradient(pcd)
# cg.initialize(x=u)
#
# frames = u0.reshape((1, Nx, Ny))
# start = time.time()
# print("solving")
# for n in range(1, nt):
#     I0 = torch.zeros_like(u, dtype=dtype, device=device)
#
#     du, dh = mdm.differentiate(u, h)
#     du = du + I0
#     b = M @ (u + dt * du)
#
#     u, total_iter = cg.solve(A, b, a_tol=1e-5, r_tol=1e-5, max_iter=100)
#     h = h + dt * dh
#
# print(f"solved in: {time.time() - start} seconds")


if __name__ == "__main__":
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
    nt = T // dt

    cfgstim1 = {'tstart': 0.0,
                'nstim': 3,
                'period': 800,
                'duration': np.max([2.0, dt]),
                'intensity': 1.0,
                'name': 'S1'}
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

    ionic_model = ModifiedMS2v()

    domain = Triangulation()
    domain.readMesh("./data/Case_1")
    domain.exportCarpFormat("atrium")
    
    # assign nodal properties
    nodal_properties = material.nodal_property_names()
    point_region_ids = domain.point_region_ids()
    npt = point_region_ids.shape[0]

    for npr in nodal_properties:
        npr_type = material.nodal_property_type(npr)
        attribute_value = ionic_model.get_attribute(npr)

        if attribute_value:
            if npr_type == "uniform":
                values = material.NodalProperty(npr, -1, -1)
            else:
                values = torch.full(size=(npt, 1), fill_value=attribute_value)
                for point_id, region_id in enumerate(point_region_ids):
                    values[point_id] = material.NodalProperty(npr, point_id, region_id)
            ionic_model.set_attribute(npr, values)

    # assemble matrix
    # matrices = Matrices(device=device, dtype=dtype)
    # K, M = matrices.assemble_matrices(triangulation, alpha)
    # K = K.to(device=device, dtype=dtype)
    # M = M.to(device=device, dtype=dtype)
    # A = M + K * dt 

    # pcd = Preconditioner()
    # pcd.create_Jocobi(A)
    


    pointlist = load_stimulus_region('./data/Case_1.vtx')  # (2168,)
    S1 = torch.zeros(size=(npt,), dtype=bool)
    S1[pointlist] = True

    # set initial conditions
    u = torch.full(size=(npt,), fill_value=0)
    h = torch.full(size=(npt,), fill_value=1)

    # cg = ConjugateGradient(pcd)
    # cg.initialize(x=u)

    # add stimulus
    nbstim = 1
    stimulus_dict = {}
    stimulus_dict[nbstim] = Stimulus(cfgstim1)
    stimulus_dict[nbstim].set_stimregion(S1)

    connectivity = domain.mesh_connectivity()
    print(connectivity)
    # solve
    # max_iter = npt // 2
    # ctime = 0
    # for i in range(nt):
    #     ctime += dt
    #     du, dh = ionic_model.differentiate(u, h)
    #     b = u + dt * du 
    #     for _, stimulus in stimulus_dict.items():
    #         I0 = stimulus.stimApp(ctime)
    #         b = b + dt * I0


