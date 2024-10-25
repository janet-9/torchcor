import torch


def apply_dirichlet_boundary_conditions(A, dirichlet_boundary_nodes):
    device = A.device

    A = A.coalesce()
    
    sparse_indices = A.indices()  # apply boundary condition
    sparse_values = A.values()

    mask = ~torch.isin(sparse_indices[0], dirichlet_boundary_nodes)
    new_indices = sparse_indices[:, mask]
    new_values = sparse_values[mask]

    identity_indices = torch.stack([dirichlet_boundary_nodes, dirichlet_boundary_nodes], dim=0).to(device=device)  # Diagonal indices
    identity_values = torch.ones_like(dirichlet_boundary_nodes)  # Diagonal values are set to 1.0

    final_indices = torch.cat([new_indices, identity_indices], dim=1)
    final_values = torch.cat([new_values, identity_values])

    return torch.sparse_coo_tensor(final_indices, final_values, A.shape)


