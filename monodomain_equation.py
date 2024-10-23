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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
print(device)


def load_stimulus_region(vtxfile):
    with open(vtxfile,'r') as fstim:
        nodes = fstim.read()
        nodes = nodes.strip().split()
    npt = int(nodes[0])
    nodes = nodes[2:]
    pointlist = -1.0 * np.ones(shape=npt, dtype=int)
    for jj, inod in enumerate(nodes):
        pointlist[jj] = int(inod)

    return pointlist.astype(int)

#
# def calculate_sigma(type, iElem, domain, properties):
#     fibres = domain.fibres[iElem, :]
#     region_id = domain.Elems()[type][iElem, -1]
#     sigma_l = properties.element_property("sigma_l", type, iElem, region_id)
#     sigma_t = properties.element_property("sigma_t", type, iElem, region_id)
#     sigma = sigma_t * torch.eye(3)
#     for i in range(3):
#         for j in range(3):
#             sigma[i, j] += (sigma_l - sigma_t) * fibres[i] * fibres[j]
#
#     return sigma
#
#
#
#
# def assign_nodal_properties(properties):
#     uniform_only = True
#     nodal_properties = properties.nodal_property_names()
#     if nodal_properties is not None:
#         point_region_ids = self._Domain.point_region_ids()
#         npt = point_region_ids.shape[0]
#         for mat_prop in nodal_properties:
#             prtype = self._materials.nodal_property_type(mat_prop)
#             refval = self.get_parameter(mat_prop)
#             if refval is not None:
#                 if prtype == 'uniform':
#                     pvals = self._materials.NodalProperty(mat_prop, -1, -1)
#                 else:
#                     uniform_only = False
#                     pvals = np.full(shape=(npt, 1), fill_value=refval.numpy())
#                     for pointID, regionID in enumerate(point_region_ids):
#                         new_val = self._materials.NodalProperty(mat_prop, pointID, regionID)
#                         pvals[pointID] = new_val
#                 self.set_parameter(mat_prop, pvals)
#     if (uniform_only or (not self._use_renumbering)):
#         self._materials.remove_all_nodal_properties()
#
# mdm = ModifiedMS2v()
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

    config = {
        'mesh_file_name': "./data/Case_1",
        'use_renumbering': True,
        'dt': dt,
        'dt_per_plot': int(1.0 / dt),  # record every ms
        'Tend': 2400}

    cfgstim1 = {'tstart': 0.0,
                'nstim': 3,
                'period': 800,
                'duration': np.max([2.0, dt]),
                'intensity': 1.0,
                'name': 'S1'}

    properties = Properties()
    properties.add_element_property("sigma_l", "uniform", diffusl)
    properties.add_element_property("sigma_t", "uniform", diffust)
    properties.add_nodal_property("tau_in", "uniform", tin)
    properties.add_nodal_property("tau_out", "uniform", tout)
    properties.add_nodal_property("tau_open", "uniform", topen)
    properties.add_nodal_property("tau_close", "uniform", tclose)
    properties.add_nodal_property("u_gate", "uniform", vg)
    properties.add_nodal_property("u_crit", "uniform", vg)
    properties.add_function('mass', lambda x: None)
    # properties.add_function('stiffness', calculate_sigma)


    pointlist = load_stimulus_region('./data/Case_1.vtx')  # (2168,)
    print(pointlist.shape)
    # S1=np.zeros(shape=model.domain().Pts().shape[0],dtype=bool)
    # S1[pointlist]=True