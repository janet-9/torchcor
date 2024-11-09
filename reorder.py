import torch
from itertools import combinations
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee


class RCM:
    def __init__(self, device, dtype):
        self.rcm_order = None
        self.rcm_map = None
        self.inverse_rcm_order = None
        self.device = device
        self.dtype = dtype

    def calculate_rcm_order(self, vertices, triangles):
        # vertices = torch.from_numpy(vertices).to(dtype=self.dtype, device=self.device)
        # triangles = torch.from_numpy(triangles).to(dtype=torch.long, device=self.device)
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

        self.rcm_order = torch.tensor(reverse_cuthill_mckee(pattern_scipy).copy(), dtype=torch.long, device=self.device)

        self.rcm_map = torch.zeros_like(self.rcm_order, dtype=torch.long, device=self.device)
        self.rcm_map[self.rcm_order] = torch.arange(n_vertices, dtype=torch.long, device=self.device)
       
        self.inverse_rcm_order = torch.argsort(self.rcm_order)

    def reorder(self, tensor_any):
        return tensor_any[self.rcm_order]

    def map(self, tensor_long):
        return self.rcm_map[tensor_long]

    def inverse(self, tensor_any):
        return tensor_any[self.inverse_rcm_order]
    

class RowDensity:
    def __init__(self, device, dtype):
        self.rcm_order = None
        self.rcm_map = None
        self.inverse_rcm_order = None
        self.device = device
        self.dtype = dtype

    def calculate_rcm_order(self, vertices, triangles):
        # vertices = torch.from_numpy(vertices).to(dtype=self.dtype, device=self.device)
        # triangles = torch.from_numpy(triangles).to(dtype=torch.long, device=self.device)
        n_vertices = vertices.shape[0]

        indices = []

        for i, _ in combinations(list(range(triangles.shape[-1])), 2):
            x = triangles[:, i]  # (N, )
            indices.append(x)  # (N, )
        
        row_indices = torch.cat(indices, dim=0)   # (N,)
        row_activity = torch.bincount(row_indices, minlength=n_vertices)

        self.rcm_order = torch.argsort(row_activity, descending=True)

        self.rcm_map = torch.zeros_like(self.rcm_order, dtype=torch.long, device=self.device)
        self.rcm_map[self.rcm_order] = torch.arange(n_vertices, dtype=torch.long, device=self.device)
       
        self.inverse_rcm_order = torch.argsort(self.rcm_order)

    def reorder(self, tensor_any):
        return tensor_any[self.rcm_order]

    def map(self, tensor_long):
        return self.rcm_map[tensor_long]

    def inverse(self, tensor_any):
        return tensor_any[self.inverse_rcm_order]
    
