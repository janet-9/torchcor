import torch


class Matrices:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype

    def shape_function_gradients(self):
        # Gradients of shape functions in the reference element (master triangle)
        return torch.tensor([[-1, -1],
                                  [1, 0],
                                  [0, 1]], device=self.device, dtype=self.dtype)

    def jacobian(self, coords):
        dN_dxi = self.shape_function_gradients()  # This should return shape (2, 3) for a single triangle

        # Assuming coords is of shape (N, 3, 2), where N is the number of triangles
        # Transpose `coords` so that it becomes (N, 2, 3) to match (2, 3) of dN_dxi
        J = torch.matmul(coords.transpose(1, 2), dN_dxi)  # Result: (N, 2, 2)

        return J

    def local_mass(self, areas):
        Me_batch = (areas / 12).unsqueeze(1).unsqueeze(2) * torch.tensor([[2, 1, 1], [1, 2, 1], [1, 1, 2]],
                                                                         device=self.device,
                                                                         dtype=self.dtype).unsqueeze(0)  # Shape (N, 3, 3)
        return Me_batch

    def local_stiffness(self, alpha, areas, coords):
        # Calculate the Jacobian for each triangle
        J_batch = self.jacobian(coords)  # Assumed to return a batch of Jacobians, shape (N, 2, 2)
        det_J_batch = torch.linalg.det(J_batch)
        inv_J_batch = torch.linalg.inv(J_batch)

        # Get shape function gradients in reference coordinates (dN/dxi)
        dN_dxi = self.shape_function_gradients()  # Assuming this is constant for all triangles, shape (3, 2)

        # Calculate dN/dxy for all triangles using the inverse Jacobian
        dN_dxy_batch = dN_dxi @ inv_J_batch  # Shape (N, 2, 3)
        # raise Exception(dN_dxy_batch[0])
        # print(alpha * det_J_batch * areas)
        Ke_batch = (0.5 * alpha * det_J_batch).view(-1, 1, 1) * dN_dxy_batch @ dN_dxy_batch .transpose(1, 2)
        # Stiffness matrix computation
        # Ke_batch = torch.zeros((coords.shape[0], 3, 3), device=self.device, dtype=self.dtype)
        # for i in range(3):
        #     for j in range(3):
        #         Ke_batch[:, i, j] = (alpha * det_J_batch * areas) * (
        #             dN_dxy_batch[:, 0, i] * dN_dxy_batch[:, 0, j] + dN_dxy_batch[:, 1, i] * dN_dxy_batch[:, 1, j])
        # raise Exception(Ke_batch[0])
        return Ke_batch

    def assemble_matrices(self, triangulation, alpha):
        # Precompute vertex coordinates for all triangles
        x_coords = torch.tensor(triangulation.x, device=self.device, dtype=self.dtype)
        y_coords = torch.tensor(triangulation.y, device=self.device, dtype=self.dtype)
        vertices = torch.tensor(triangulation.triangles, device=self.device, dtype=torch.long)

        # Get the coordinates for all triangles
        coords = torch.stack([x_coords[vertices], y_coords[vertices]],
                             dim=2)  # Shape: (N, 3, 2), where N is number of triangles

        # Augment with a column of ones for determinant calculation (for areas)
        coords_augmented = torch.cat([torch.ones(vertices.shape[0], 3, 1, device=self.device, dtype=self.dtype), coords], dim=2)
        # Calculate areas for all triangles at once
        areas = 0.5 * torch.abs(torch.linalg.det(coords_augmented))

        # Mass and stiffness matrix for all triangles: (N, 3, 3)
        Me_batch = self.local_mass(areas)
        Ke_batch = self.local_stiffness(alpha, areas, coords)

        # Lists to store the indices and values of non-zero elements for K and M
        rows, cols, K_vals, M_vals = [], [], [], []
        # Collect contributions for global matrices in sparse form
        for i in range(3):
            for j in range(3):
                rows.extend(vertices[:, i].tolist())
                cols.extend(vertices[:, j].tolist())
                K_vals.extend(Ke_batch[:, i, j].tolist())
                M_vals.extend(Me_batch[:, i, j].tolist())

        # Create sparse tensors from the accumulated lists
        npoints = len(triangulation.x)  # Total number of points in the mesh
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




