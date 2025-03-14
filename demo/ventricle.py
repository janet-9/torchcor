import torchcor as tc
from torchcor.simulator import Monodomain
from torchcor.ionic import TenTusscherPanfilov
from pathlib import Path

tc.set_device("cuda:0")
simulation_time = 500
dt = 0.01

home_dir = Path.home()

ionic_model = TenTusscherPanfilov(cell_type="ENDO", dt=dt)
simulator = Monodomain(ionic_model, T=simulation_time, dt=dt)
simulator.load_mesh(path=f"{home_dir}/Data/ventricle/")
simulator.add_condutivity([34, 35], il=0.5272, it=0.2076, el=1.0732, et=0.4227)
simulator.add_condutivity([44, 45, 46], il=0.9074, it=0.3332, el=0.9074, et=0.3332)

simulator.add_stimulus(f"{home_dir}/Data/ventricle/LV_sf.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(f"{home_dir}/Data/ventricle/LV_pf.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(f"{home_dir}/Data/ventricle/LV_af.vtx", start=0.0, duration=1.0, intensity=100)
simulator.add_stimulus(f"{home_dir}/Data/ventricle/RV_sf.vtx", start=5.0, duration=1.0, intensity=100)
simulator.add_stimulus(f"{home_dir}/Data/ventricle/RV_mod.vtx", start=5.0, duration=1.0, intensity=100)

simulator.solve(a_tol=1e-5, 
                r_tol=1e-5, 
                max_iter=100, 
                linear_guess=True,
                plot_interval=10, 
                verbose=True, 
                format="")

