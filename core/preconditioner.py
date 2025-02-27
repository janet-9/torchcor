import torch


@torch.jit.script
class Preconditioner:
    def __init__(self):
        self.PC = torch.tensor([1.0])

    def create_Jocobi(self, A):
        device, dtype = A.device, A.dtype

        A = A.coalesce()

        sparse_indices = A.indices()
        sparse_values = A.values()

        row_indices = sparse_indices[0, :]
        col_indices = sparse_indices[1, :]

        # Create a mask for diagonal elements (where row == col)
        diagonal_mask = row_indices == col_indices

        # Extract the diagonal values
        PC = torch.zeros(A.shape[0], device=device, dtype=dtype)
        PC[row_indices[diagonal_mask]] = sparse_values[diagonal_mask]

        self.PC = 1.0 / PC

        # raise Exception(A[sparse_indices[0][diagonal_mask[1000]], sparse_indices[1][1000]], PC[1000])

    # def create_scaled_Jacobi(self, A):
    #     self.create_Jocobi(A)
    #
    #     row_norms = torch.sqrt(torch.sparse.sum(A ** 2, dim=1).to_dense())
    #     row_norms[row_norms == 0] = 1.0
    #
    #     self.PC = self.PC / row_norms


    def apply(self, r):
        z = self.PC * r
        
        return z
    

    # def create_diagonal_matrix(self, A):
    #     device, dtype = A.device, A.dtype
    #
    #     sparse_indices = A._indices()
    #     sparse_values = A._values()
    #
    #     diagonal_mask = sparse_indices[0, :] == sparse_indices[1, :]
    #
    #     diagonal_values = sparse_values[diagonal_mask]
    #     diagonal_indices = sparse_indices[:, diagonal_mask]
    #
    #     D = torch.sparse_coo_tensor(diagonal_indices, diagonal_values, A.shape).to(device=device, dtype=dtype)
    #
    #     return D
