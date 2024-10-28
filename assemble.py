import torch
from itertools import combinations
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee


class Matrices2D:
    def __init__(self, vertices, triangles, device, dtype):
        self.vertices = torch.tensor(vertices, dtype=torch.float64, device=device) # (12100, 2)
        self.n_vertices = vertices.shape[0]
        self.triangles = torch.tensor(triangles, dtype=torch.long, device=device) # (23762, 3)
        self.device = device
        self.dtype = dtype

    def shape_function_gradients(self):
        # Gradients of shape functions in the reference element (master triangle) (3, 2)
        return torch.tensor([[-1, -1],
                                  [1, 0],
                                  [0, 1]], device=self.device, dtype=self.dtype)

    def jacobian(self, triangle_coords):
        dN_dxi = self.shape_function_gradients()

        # Transpose `coords` (N, 3, 2) so that it becomes (N, 2, 3) to match (3, 2) of dN_dxi
        J = triangle_coords.transpose(1, 2) @ dN_dxi  # Result: (N, 2, 2)

        return J

    def local_mass(self, areas):
        Me_batch = (areas / 12).unsqueeze(1).unsqueeze(2) * torch.tensor([[2, 1, 1], [1, 2, 1], [1, 1, 2]],
                                                                         device=self.device,
                                                                         dtype=self.dtype).unsqueeze(0)  # Shape (N, 3, 3)
        return Me_batch

    def local_stiffness(self, alpha, triangle_coords):
        # Calculate the Jacobian for each triangle
        J_batch = self.jacobian(triangle_coords)  # Assumed to return a batch of Jacobians, shape (N, 2, 2)
        det_J_batch = torch.linalg.det(J_batch)
        inv_J_batch = torch.linalg.inv(J_batch)

        # Calculate dN/dxy for all triangles using the inverse Jacobian
        dN_dxi = self.shape_function_gradients()  # Assuming this is constant for all triangles, shape (3, 2)
        dN_dxy_batch = dN_dxi @ inv_J_batch  # Shape (N, 3, 2)
        Ke_batch = (0.5 * alpha * det_J_batch).view(-1, 1, 1) * dN_dxy_batch @ dN_dxy_batch.transpose(1, 2)

        return Ke_batch
    
    def calculate_area(self, triangle_coords):
        coords_augmented = torch.cat([torch.ones(triangle_coords.shape[0], 3, 1, device=self.device, dtype=self.dtype), triangle_coords], dim=2)
        areas = 0.5 * torch.abs(torch.linalg.det(coords_augmented))

        return areas
    
    def construct_local_matrices(self, alpha):
        triangle_coords = self.vertices[self.triangles]  # (23762, 3, 2)

        # Calculate areas for all triangles at once
        areas = self.calculate_area(triangle_coords)

        # Mass and stiffness matrix for all triangles: (N, 3, 3)
        Me_batch = self.local_mass(areas)
        Ke_batch = self.local_stiffness(alpha, triangle_coords)

        return Me_batch, Ke_batch

    def assemble_matrices(self, alpha):
        # Precompute vertex coordinates for all triangles
        Me_batch, Ke_batch = self.construct_local_matrices(alpha)
        # Lists to store the indices and values of non-zero elements for K and M
        rows, cols, K_vals, M_vals = [], [], [], []
        # Collect contributions for global matrices in sparse form
        for i in range(3):
            for j in range(3):
                rows.extend(self.triangles[:, i].tolist())
                cols.extend(self.triangles[:, j].tolist())
                K_vals.extend(Ke_batch[:, i, j].tolist())
                M_vals.extend(Me_batch[:, i, j].tolist())

        # Create sparse tensors from the accumulated lists
        K = torch.sparse_coo_tensor(
            indices=[rows, cols],
            values=K_vals,
            size=(self.n_vertices, self.n_vertices)
        )

        M = torch.sparse_coo_tensor(
            indices=[rows, cols],
            values=M_vals,
            size=(self.n_vertices, self.n_vertices)
        )

        K = K.to(device=self.device, dtype=self.dtype)
        M = M.to(device=self.device, dtype=self.dtype)

        return K.coalesce(), M.coalesce()
    
    def renumber_permutation(self):
        indices = []

        for i, j in combinations(list(range(self.triangles.shape[-1])), 2):
            x = self.triangles[:, i]  # (N, )
            y = self.triangles[:, j]  # (N, )
            indices.append(torch.stack([x, y],  dim=0))  # (2, N)
            # TODO: add diagonal? 
        
        indices = torch.cat(indices, dim=1)   # (2, N)
        values = torch.ones(indices.shape[1])
        # pattern = torch.sparse_coo_tensor(indices, 
        #                                   values, 
        #                                   size=(self.n_vertices, self.n_vertices),
        #                                   device=self.device)
        
        indices = indices.cpu().numpy()
        values = values.cpu().numpy()
        pattern_scipy = sp.csr_matrix((values, (indices[0], indices[1])), shape=(self.n_vertices, self.n_vertices))

        rcm_order = reverse_cuthill_mckee(pattern_scipy)
        # raise Exception(rcm_order)
    
        return rcm_order
        

        


class Matrices3DSurface(Matrices2D):
    def __init__(self, vertices, triangles, device, dtype):
        super().__init__(vertices, triangles, device, dtype)

    def jacobian(self, triangle_coords):
        dN_dxi = self.shape_function_gradients()
        # Transpose `coords` to match dimensions: (N, 3, 3)

        u1 = triangle_coords[:, 1, :] - triangle_coords[:, 0, :]  # (N, 3)
        u2 = triangle_coords[:, 2, :] - triangle_coords[:, 0, :]  # (N, 3)

        alpha = u1 / torch.linalg.norm(u1, dim=1, keepdim=True)  # (N, 3)

        beta = u2 - torch.sum(u2 * alpha, dim=1, keepdim=True) * alpha  # (N, 3)
        beta = beta / torch.linalg.norm(beta, dim=1, keepdim=True)  # (N, 3)

        rotation_matrix = torch.stack([alpha, beta], dim=1)  # (N, 2, 3)

        J = rotation_matrix @ triangle_coords.transpose(1, 2) @ dN_dxi

        return J
    
    def calculate_area(self, triangle_coords):
        # Calculate the area of the triangle using cross product
        a = triangle_coords[:, 1] - triangle_coords[:, 0]
        b = triangle_coords[:, 2] - triangle_coords[:, 0]
        cross_product = torch.linalg.cross(a, b)  # Shape (N, 3)
        areas = 0.5 * torch.norm(cross_product, dim=1)  # Shape (N,)

        return areas
