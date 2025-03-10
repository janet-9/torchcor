import pygmsh
import numpy as np
from pathlib import Path

scale = 1000

for mesh_size in [0.3, 0.2, 0.15, 0.12, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05]:
    print(mesh_size)
    with pygmsh.geo.Geometry() as geom:
        geom.add_box(
            x0=0, x1=20,  
            y0=0, y1=7,  
            z0=0, z1=3, 
            mesh_size=mesh_size,
        )
        mesh = geom.generate_mesh(dim=3)

        nodes = mesh.points * scale
        n_nodes = nodes.shape[0]
        elems = mesh.cells_dict['tetra']

        folder = Path(f"{Path.home()}/Data/volume/{n_nodes}")
        folder.mkdir(parents=True, exist_ok=True)

        pts_file = folder / f"{0}.pts"
        with pts_file.open('w') as f:
            f.write(f"{nodes.shape[0]}\n")
            for point in nodes:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

        elem_file = folder / f"{0}.elem"
        with elem_file.open('w') as f:
            f.write(f"{elems.shape[0]}\n")
            for elem in elems:
                f.write(f"Tt {elem[0]} {elem[1]} {elem[2]} {elem[3]} 0\n")
        
        lon_file = folder / f"{0}.lon"
        with lon_file.open('w') as f:
            f.write(f"{1}\n")
            for _ in range(elems.shape[0]):
                f.write(f"1 0 0\n")
        
        stim_region = np.where((nodes[:, 0] <= 1.5 * scale) &  
                                (nodes[:, 1] <= 1.5 * scale) & 
                                (nodes[:, 2] <= 1.5 * scale)   
                                )[0] 
        vtx_file = folder / f"{0}.vtx"
        with vtx_file.open('w') as f:
            f.write(f"{len(stim_region)}\n")
            f.write(f"extra\n")
            for region_id in stim_region:
                f.write(f"{region_id}\n")