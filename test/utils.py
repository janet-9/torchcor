import numpy as np
import torch

def load_stimulus_region(vtxfile: str) -> np.ndarray:
    """ load_stimulus_region(vtxfile) reads the file vtxfile to
    extract point IDs where stimulus will be applied
    """
    with open(vtxfile, 'r') as f:
        nodes = f.read()
        nodes = nodes.strip().split()
    
    n_nodes = int(nodes[0])
    nodes = nodes[2:]
    pointlist = -1.0 * np.ones(shape=n_nodes, dtype=int)
    for i, node in enumerate(nodes):
        pointlist[i] = int(node)
    
    return pointlist.astype(int)

def save_coo_matrix(matrix, filename):
    coo_data = {
        'row': matrix.indices()[0].cpu(),  # Make sure they are on CPU
        'col': matrix.indices()[1].cpu(),
        'data': matrix.values().cpu(),      # Ensure values are on CPU
        'shape': matrix.shape
    }
    torch.save(coo_data, filename)