import numpy as np
from mesh.triangulation import Triangulation
from mesh.materialproperties import MaterialProperties

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