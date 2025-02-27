import numpy as np
import torch


def save_coo_matrix(matrix, filename):
    coo_data = {
        'row': matrix.indices()[0].cpu(),  # Make sure they are on CPU
        'col': matrix.indices()[1].cpu(),
        'data': matrix.values().cpu(),      # Ensure values are on CPU
        'shape': matrix.shape
    }
    torch.save(coo_data, filename)