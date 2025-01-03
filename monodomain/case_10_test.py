import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from core.assemble import Matrices3DSurface
from core.preconditioner import Preconditioner
from core.solver import ConjugateGradient
from core.visualize import VTK3DSurface, VTK3DSurfaceSave
from core.reorder import RCM as RCM
import time
from mesh.triangulation import Triangulation
from mesh.materialproperties import MaterialProperties
from mesh.stimulus import Stimulus
from monodomain.tools import load_stimulus_region
import numpy as np
from mesh.igbreader import IGBReader


class AtriumSimulatorCourtemanche:
    def __init__(self, ionic_model, T, dt, apply_rcm, device=None, dtype=torch.float64):
        self.device = torch.device(device) if device is not None else "cuda:0" \
            if torch.cuda.is_available() else "cpu"
        self.dtype = dtype

        self.T = T  # ms = 2.4s
        self.dt = dt  # ms
        self.nt = int(T / dt)
        self.rcm = RCM(device=device, dtype=dtype) if apply_rcm else None

        self.ionic_model = ionic_model
        self.ionic_model.construct_tables()

        self.pcd = None

        self.point_region_ids = None
        self.n_nodes = None
        self.vertices = None
        self.triangles = None
        self.fibers = None

        self.material_config = None

        self.K = None
        self.M = None
        self.A = None

        self.stimulus_region = None
        self.stimuli = []

    def load_mesh(self, path="/Users/bei/Project/FinitePDE/data/Case_1"):
        mesh = Triangulation()
        mesh.readMesh(path)

        self.point_region_ids = mesh.point_region_ids()
        self.n_nodes = self.point_region_ids.shape[0]

        self.vertices = torch.from_numpy(mesh.Pts()).to(dtype=self.dtype, device=self.device)
        self.triangles = torch.from_numpy(mesh.Elems()['Trias'][:, :-1]).to(dtype=torch.long, device=self.device)
        self.fibers = torch.from_numpy(mesh.Fibres()).to(dtype=self.dtype, device=self.device)

        if self.rcm is not None:
            self.rcm.calculate_rcm_order(self.vertices, self.triangles)

        reader = IGBReader()
        reader.read("/home/bzhou6/Projects/TorchCor/monodomain/vm_300.igb")
        carp_solution = reader.data()

        cor_solution = np.load("/home/bzhou6/Projects/TorchCor/monodomain/Case_10_300_1.npy")

        for i in range(301):
            saver = VTK3DSurfaceSave(self.vertices.cpu(), self.triangles.cpu())
            saver.save_frame(carp_values=carp_solution[i],
                             cor_values=cor_solution[i],
                             frame_path=f"./vtk_files_300_1/frame_{i}.vtk")
        


if __name__ == "__main__":
    from ionic import CourtemancheRamirezNattel
    import torch
    from pathlib import Path

    simulation_time = 1
    dt = 0.01
    stim_config = {'tstart': 0.0,
                'nstim': 3,
                'period': 800,
                'duration': 2.0,
                'intensity': 100.0,
                'name': 'S1'}
    material_config = {"diffusl": 0.4 * 1000 * 100,
                       "diffust": 0.4 * 1000 * 100}


    device = torch.device(f"cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # dtype = torch.float32
    dtype = torch.float64

    home_directory = Path.home()
    ionic_model = CourtemancheRamirezNattel(dt=dt, device=device, dtype=dtype)
    print(ionic_model.default_parameters())
    ionic_model.reset_parameters()
    simulator = AtriumSimulatorCourtemanche(ionic_model, T=simulation_time, dt=dt, apply_rcm=False, device=device, dtype=dtype)
    simulator.load_mesh(path=f"{home_directory}/Data/atrium/Case_10/Case_10")
    # simulator.add_material_property(material_config)
    # simulator.set_stimulus_region(path=f"{home_directory}/Data/atrium/Case_10/Case_10.vtx")
    # simulator.add_stimulus(stim_config)
    # simulator.assemble()
    # simulator.solve(a_tol=1e-5, r_tol=1e-5, max_iter=1000, plot_interval=0.01, verbose=True)


