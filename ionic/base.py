import torch
from collections import OrderedDict


class BaseCellModel:
    def __init__(self, cell_model, device, dtype=torch.float64):
        self.cell_model = cell_model
        self.device = device
        self.dtype = dtype

        init_states, init_constants = self.cell_model.initConsts()
        self.states = torch.tensor(init_states, device=device, dtype=dtype)
        self.constants = None

        self.name_constant_dict = OrderedDict()
        (_, _, _, legend_constants) = self.cell_model.createLegends()
        for legend_constant, init_constant in zip(legend_constants, init_constants):
            constant_name = legend_constant.split()[0]
            self.name_constant_dict[constant_name] = init_constant

        self.H = None
        self.dt = None

    def default_constants(self):
        return dict(self.name_constant_dict)

    def reset_constant(self, name, value):
        self.name_constant_dict[name] = value

    def initialize(self, n_nodes, dt):
        self.states = self.states.repeat(n_nodes, 1).clone()

        self.constants = torch.tensor(list(self.name_constant_dict.values()), device=self.device, dtype=self.dtype)
        U = self.states[:, 0].clone()
        self.H = self.states[:, 1:].clone()
        self.dt = dt

        return U

    def differentiate(self, U):
        self.states[:, 0] = U
        self.states[:, 1:] = self.H

        rates = self.compute_rates(states=self.states, constants=self.constants)
        self.dt_ionic = self.dt / 100
        for _ in range(100):
            dH = rates[:, 1:]
            self.H += self.dt_ionic * dH
            self.states[:, 1:] = self.H
            rates = self.compute_rates(states=self.states, constants=self.constants)

        dU = rates[:, 0]

        return dU

    def compute_rates(self, states, constants):
        rates = torch.zeros_like(states)
        algebraic = torch.zeros((states.shape[0], self.cell_model.sizeAlgebraic), device=self.device, dtype=self.dtype)


        return rates

    def set_attribute(self, name, value):
        setattr(self, name, value)

    def get_attribute(self, name: str):
        return getattr(self, name, None)