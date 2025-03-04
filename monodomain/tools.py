import numpy as np
from mesh.triangulation import Triangulation
from mesh.materialproperties import MaterialProperties

def load_stimulus_region(vtxfile: str) -> np.ndarray:
    with open(vtxfile, 'r') as f:
        nodes = f.read()
        nodes = nodes.strip().split()
    
    n_nodes = int(nodes[0])
    nodes = nodes[2:]
    pointlist = -1.0 * np.ones(shape=n_nodes, dtype=np.int32)
    for i, node in enumerate(nodes):
        pointlist[i] = int(node)
    
    return pointlist.astype(np.int32)


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