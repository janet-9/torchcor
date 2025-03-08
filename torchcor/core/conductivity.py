import torch


class Conductivity:
    def __init__(self, regions, dtype):
        self.regions = regions
        self.dtype = dtype

        self.sigma_l = torch.zeros_like(self.regions, dtype=self.dtype)
        self.sigma_t = torch.zeros_like(self.regions, dtype=self.dtype)
    
    def add(self, region_ids, il, it, el=None, et=None):
        if el == None and et == None:
            l = il
            t = it
        else:
            l = il * el * (1 / (il + el))
            t = it * et * (1 / (it + et))

        for id in region_ids:
            mask = self.regions == id
            self.sigma_l[mask] = l
            self.sigma_t[mask] = t


    def calculate_sigma(self, fibres):
        sigma_l = self.sigma_l.view(self.sigma_l.shape[0], 1, 1)
        sigma_t = self.sigma_t.view(self.sigma_t.shape[0], 1, 1)
        sigma = sigma_t * torch.eye(3, device=fibres.device, dtype=self.dtype).unsqueeze(0).expand(fibres.shape[0], 3, 3)
        sigma += (sigma_l - sigma_t) * fibres.unsqueeze(2) @ fibres.unsqueeze(1)

        return sigma