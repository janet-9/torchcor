import torch


class Preconditioner:
    def __init__(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.PC = None

    def create_Jocobi(self, A):
        sparse_indices = A._indices()
        sparse_values = A._values()

        PC = torch.zeros(A.shape[0], device=self.device, dtype=self.dtype)
        for i in range(sparse_indices.shape[1]):
            row, col = sparse_indices[:, i]
            if row == col:  # Check if it's a diagonal element
                PC[row] = sparse_values[i]

        self.PC = PC

    def apply(self, r):
        z = (1 / self.PC) * r
        # z = torch.linalg.solve(PC, r)
        return z