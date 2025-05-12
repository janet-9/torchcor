import torch


class Conductivity:
    def __init__(self, regions, dtype):
        self.regions = regions
        self.dtype = dtype

        self.sigma_il = torch.zeros_like(self.regions, dtype=self.dtype)
        self.sigma_it = torch.zeros_like(self.regions, dtype=self.dtype)

        self.sigma_el = torch.zeros_like(self.regions, dtype=self.dtype)
        self.sigma_et = torch.zeros_like(self.regions, dtype=self.dtype)

        self.sigma_l = torch.zeros_like(self.regions, dtype=self.dtype)
        self.sigma_t = torch.zeros_like(self.regions, dtype=self.dtype)
    
    def add(self, region_ids, il, it, el=None, et=None):
        if el is None and et is None:
            self.sigma_il = None
            self.sigma_it = None
            self.sigma_el = None
            self.sigma_et = None
            
            l = il
            t = it
            for id in region_ids:
                mask = self.regions == id
                
                self.sigma_l[mask] = l
                self.sigma_t[mask] = t

        else:
            l = il * el * (1 / (il + el))
            t = it * et * (1 / (it + et))

            for id in region_ids:
                mask = self.regions == id

                self.sigma_il[mask] = il
                self.sigma_it[mask] = it

                self.sigma_el[mask] = el
                self.sigma_et[mask] = et

                self.sigma_l[mask] = l
                self.sigma_t[mask] = t

    def calculate_sigma(self, fibres):
        if self.sigma_il is None and self.sigma_it is None and self.sigma_el is None and self.sigma_et is None:
            sigma_i = None
            sigma_e = None
        else:
            sigma_il = self.sigma_il.view(self.sigma_l.shape[0], 1, 1)
            sigma_it = self.sigma_it.view(self.sigma_t.shape[0], 1, 1)
            sigma_i = sigma_it * torch.eye(3, device=fibres.device, dtype=self.dtype).unsqueeze(0).expand(fibres.shape[0], 3, 3)
            sigma_i += (sigma_il - sigma_it) * fibres.unsqueeze(2) @ fibres.unsqueeze(1)

            sigma_el = self.sigma_el.view(self.sigma_l.shape[0], 1, 1)
            sigma_et = self.sigma_et.view(self.sigma_t.shape[0], 1, 1)
            sigma_e = sigma_et * torch.eye(3, device=fibres.device, dtype=self.dtype).unsqueeze(0).expand(fibres.shape[0], 3, 3)
            sigma_e += (sigma_el - sigma_et) * fibres.unsqueeze(2) @ fibres.unsqueeze(1)

        sigma_l = self.sigma_l.view(self.sigma_l.shape[0], 1, 1)
        sigma_t = self.sigma_t.view(self.sigma_t.shape[0], 1, 1)
        sigma_m = sigma_t * torch.eye(3, device=fibres.device, dtype=self.dtype).unsqueeze(0).expand(fibres.shape[0], 3, 3)
        sigma_m += (sigma_l - sigma_t) * fibres.unsqueeze(2) @ fibres.unsqueeze(1)

        return sigma_i, sigma_e, sigma_m
    

