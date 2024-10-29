import torch
from itertools import combinations
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee


class RCM:
    def __init__(self):
        self.rcm_order = None
        self.inverse_rcm_order = None

    def calculate_rcm_order(self, vertices, triangles):
        vertices = torch.from_numpy(vertices).to(dtype=torch.float64)
        triangles = torch.from_numpy(triangles).to(dtype=torch.long)
        n_vertices = vertices.shape[0]

        indices = []

        for i, j in combinations(list(range(triangles.shape[-1])), 2):
            x = triangles[:, i]  # (N, )
            y = triangles[:, j]  # (N, )
            indices.append(torch.stack([x, y],  dim=0))  # (2, N)
        
        indices = torch.cat(indices, dim=1)   # (2, N)
        values = torch.ones(indices.shape[1])

        indices = indices.cpu().numpy()
        values = values.cpu().numpy()
        pattern_scipy = sp.csr_matrix((values, (indices[0], indices[1])), shape=(n_vertices, n_vertices))

        rcm = torch.tensor(reverse_cuthill_mckee(pattern_scipy).copy(), dtype=torch.long)

        self.rcm_order = torch.zeros_like(rcm)
        self.rcm_order[rcm] = torch.arange(n_vertices)
       
        self.inverse_rcm_order = torch.argsort(rcm)  

        return vertices[rcm], self.apply(triangles)

    def apply(self, tensor):
        return self.rcm_order[tensor]

    def inverse(self, tensor):
        return tensor[self.inverse_rcm_order]
    

