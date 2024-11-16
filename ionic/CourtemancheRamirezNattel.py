import torch
import cellml.courtemanche_ramirez_nattel_1998 as cell_model
from base import BaseCellModel

class CourtemancheRamirezNattel(BaseCellModel):
    def __init__(self, cell_model, device, dtype):
        super().__init__(cell_model, device, dtype)

    def compute_rates(self, states, constants):
        rates = torch.zeros_like(states)
        algebraic = torch.zeros((states.shape[0], self.cell_model.sizeAlgebraic), device=self.device, dtype=self.dtype)



        return rates