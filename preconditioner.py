import torch


class Preconditioner:
    def __init__(self):
        self.PC = None

    def create_Jocobi(self, A):
        device, dtype = A.device, A.dtype

        sparse_indices = A._indices()
        sparse_values = A._values()

        row_indices = sparse_indices[0, :]
        col_indices = sparse_indices[1, :]

        # Create a mask for diagonal elements (where row == col)
        diagonal_mask = row_indices == col_indices

        # Extract the diagonal values
        PC = torch.zeros(A.shape[0], device=device, dtype=dtype)
        PC[row_indices[diagonal_mask]] = sparse_values[diagonal_mask]

        self.PC = PC

    def apply(self, r):
        z = (1 / self.PC) * r
        # z = torch.linalg.solve(PC, r)
        return z