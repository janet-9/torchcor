from pathlib import Path
import numpy as np
np.set_printoptions(precision=15, suppress=False)

class MeshReader:
    def __init__(self, mesh_dir: str):
        self.mesh_dir = Path(mesh_dir)

        self.node_file: Path = None
        self.elem_file: Path = None
        self.fibre_file: Path = None
        
        for file_path in self.mesh_dir.iterdir():
            if file_path.suffix == ".pts":
                self.node_file = file_path
            elif file_path.suffix == ".elem":
                self.elem_file = file_path
            elif file_path.suffix == ".lon":
                self.fibre_file = file_path

        self.nodes: np.array = None
        self.elems: np.array = None
        self.regions: np.array = None
        self.fibres: np.array = None

    def read_nodes(self, unit_conversion=1000):
        # process the header
        with self.node_file.open("r") as f:
            num_pts_expected = int(f.readline().strip()) 
        
        # read the data
        nodes = np.loadtxt(self.node_file, dtype=float, skiprows=1)
        num_pts_actual = nodes.shape[0]
        if num_pts_actual != num_pts_expected:
            raise ValueError(f"Mismatch in number of nodes: expected {num_pts_expected}, but found {num_pts_actual}")

        self.nodes = nodes / unit_conversion
    
    def read_elems(self):
        with self.elem_file.open("r") as f:
            num_elems_expected = int(f.readline().strip())
            first_data_line = f.readline().strip().split()
            usecols = list(range(1, len(first_data_line)))

        data = np.loadtxt(self.elem_file, dtype=int, skiprows=1, usecols=usecols)
        
        num_elems_actual = data.shape[0]
        if num_elems_actual != num_elems_expected:
            raise ValueError(f"Mismatch: expected {num_elems_expected}, but found {num_elems_actual}")

        self.elems = data[:, :-1]
        self.regions = data[:, -1]

    def read_fibres(self):
        fibres = np.loadtxt(self.fibre_file, dtype=np.float64, skiprows=1)
        self.fibres = fibres
    
    def read(self, unit_conversion=1000):
        self.read_nodes(unit_conversion)
        self.read_elems()
        self.read_fibres()

        return self.nodes, self.elems, self.regions, self.fibres

    def compute_edges(self):
        pass

if __name__ == "__main__":
    import time
    start_time = time.time()
    # reader = MeshReader("/home/bzhou6/Data/atrium/Case_1")
    reader = MeshReader("/home/bzhou6/Data/ventricle/")
    reader.read_nodes()
    reader.read_elems()
    reader.read_fibres()
    print(time.time() - start_time)

    print(reader.nodes[0], reader.elems[0], reader.regions[0], reader.fibres[0])