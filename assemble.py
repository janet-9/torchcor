import torch


class Matrices2D:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def shape_function_gradients(self):
        # Gradients of shape functions in the reference element (master triangle) (3, 2)
        return torch.tensor([[-1, -1],
                                  [1, 0],
                                  [0, 1]], device=self.device, dtype=self.dtype)

    def jacobian(self, triangle_coords):
        dN_dxi = self.shape_function_gradients()
        # Transpose `coords` so that it becomes (N, 2, 3) to match (3, 2) of dN_dxi
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
    
    def construct_local_matrices(self, vertices, triangles, alpha):
        vertices = torch.tensor(vertices, dtype=torch.float64)  # (12100, 2)
        triangles = torch.tensor(triangles, dtype=torch.long)   # (23762, 3)

        triangle_coords = vertices[triangles]  # (23762, 3, 2)

        # Calculate areas for all triangles at once
        areas = self.calculate_area(triangle_coords)

        # Mass and stiffness matrix for all triangles: (N, 3, 3)
        Me_batch = self.local_mass(areas)
        Ke_batch = self.local_stiffness(alpha, triangle_coords)

        return Me_batch, Ke_batch

    def assemble_matrices(self, vertices, triangles, alpha):
        # Precompute vertex coordinates for all triangles
        Me_batch, Ke_batch = self.construct_local_matrices(vertices, triangles, alpha)
        # Lists to store the indices and values of non-zero elements for K and M
        rows, cols, K_vals, M_vals = [], [], [], []
        # Collect contributions for global matrices in sparse form
        for i in range(3):
            for j in range(3):
                rows.extend(triangles[:, i].tolist())
                cols.extend(triangles[:, j].tolist())
                K_vals.extend(Ke_batch[:, i, j].tolist())
                M_vals.extend(Me_batch[:, i, j].tolist())

        # Create sparse tensors from the accumulated lists
        npoints = len(vertices)  # Total number of points in the mesh
        K = torch.sparse_coo_tensor(
            indices=[rows, cols],
            values=K_vals,
            size=(npoints, npoints)
        )

        M = torch.sparse_coo_tensor(
            indices=[rows, cols],
            values=M_vals,
            size=(npoints, npoints)
        )

        return K, M


class Matrices3DSurface(Matrices2D):
    def __init__(self, device, dtype):
        super().__init__(device, dtype)

    def jacobian(self, triangle_coords):
        dN_dxi = self.shape_function_gradients()
        # Transpose `coords` to match dimensions: (N, 3, 2) becomes (N, 2, 3)
        J = triangle_coords[:, :, :2].transpose(1, 2) @ dN_dxi  # Result: (N, 2, 2)

        return J
    
    def calculate_area(self, triangle_coords):
        # Calculate the area of the triangle using cross product
        a = triangle_coords[:, 1] - triangle_coords[:, 0]
        b = triangle_coords[:, 2] - triangle_coords[:, 0]
        cross_product = torch.cross(a, b)  # Shape (N, 3)
        areas = 0.5 * torch.norm(cross_product, dim=1)  # Shape (N,)

        return areas
    
