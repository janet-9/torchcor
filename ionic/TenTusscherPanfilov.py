import torch
from math import log, exp, expm1


@torch.jit.script
class TenTusscherPanfilov:
    def __init__(self, dt: float, device: torch.device, dtype: torch.dtype = torch.float64):


        self.dt = dt
        self.device = device
        self.dtype = dtype

    def interpolate(self, X, table, mn: float, mx: float, res: float, step: float, mx_idx: int):
        X = torch.clamp(X, mn, mx)
        idx = ((X - mn) * step).to(torch.long)
        lower_idx = torch.clamp(idx, 0, mx_idx - 1)
        higher_idx = lower_idx + 1
        lower_pos = lower_idx * res + mn
        w = ((X - lower_pos) / res).unsqueeze(1)
        return (1 - w) * table[lower_idx] + w * table[higher_idx]

    def construct_tables(self):
        pass