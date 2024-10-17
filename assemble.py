import torch


def assemble_matrices(triangulation, alpha):
    npoints = len(triangulation.x)  # Total number of points in the mesh

    # Lists to store the indices and values of non-zero elements for K and M
    rows, cols, K_vals, M_vals = [], [], [], []

    for element in triangulation.triangles:
        # Get the coordinates of the three vertices of the triangle
        vertices = element[:]
        x_coords = triangulation.x[vertices]
        y_coords = triangulation.y[vertices]

        # Compute area of the triangle (used in both stiffness and mass matrices)
        area = 0.5 * abs(
            x_coords[0] * (y_coords[1] - y_coords[2]) +
            x_coords[1] * (y_coords[2] - y_coords[0]) +
            x_coords[2] * (y_coords[0] - y_coords[1])
        )
        # Local stiffness matrix (based on gradients of linear basis functions)
        Ke = (alpha / (4 * area)) * torch.tensor([[2, -1, -1],
                                                  [-1, 2, -1],
                                                  [-1, -1, 2]])

        # Local mass matrix (based on linear basis functions)
        Me = (area / 12) * torch.tensor([[2, 1, 1],
                                         [1, 2, 1],
                                         [1, 1, 2]])

        # Add local contributions to the global matrices
        for i in range(3):
            for j in range(3):
                # Store K and M values in sparse form (row, col, value)
                rows.append(vertices[i])
                cols.append(vertices[j])
                K_vals.append(Ke[i, j])
                M_vals.append(Me[i, j])

    # Create sparse tensors from the accumulated lists
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