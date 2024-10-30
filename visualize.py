import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import pyvista as pv
from pathlib import Path


class Visualization2D:
    def __init__(self, frames, vertices, triangles, dt, ts_per_frame):
        # Convert frames to numpy if they are in a tensor format
        self.frames = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
        self.vertices = vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices
        self.triangles = triangles.cpu().numpy() if isinstance(triangles, torch.Tensor) else triangles  # Assuming triangles are already in the right format
        self.dt = dt
        self.ts_per_frame = ts_per_frame
        self.nt = len(self.frames)

    def plotheatmap(self, k):
        plt.clf()

        plt.title(f"Temperature at t = {k * self.dt * self.ts_per_frame:.3f} unit time")
        plt.xlabel("x")
        plt.ylabel("y")
        
        # Use the triangles to create a heatmap
        plt.tripcolor(self.vertices[:, 0], 
                      self.vertices[:, 1], 
                      self.triangles, 
                      self.frames[k].flatten(),
                      cmap=plt.cm.jet, vmin=0, vmax=100, shading='gouraud')
        plt.colorbar()

    def animate(self, k):
        self.plotheatmap(k)

    def save_gif(self, filepath):
        fig = plt.figure()
        anim = animation.FuncAnimation(fig, self.animate, interval=1, frames=self.nt, repeat=False)
        anim.save(filepath)  


class Visualization3DSurface:
    def __init__(self, frames, vertices, triangles, dt, ts_per_frame):
        # Convert frames to numpy if they are in a tensor format
        self.frames = frames.cpu().numpy() if isinstance(frames, torch.Tensor) else frames
        self.vertices = vertices.cpu().numpy() if isinstance(vertices, torch.Tensor) else vertices
        self.triangles = triangles.cpu().numpy() if isinstance(triangles, torch.Tensor) else triangles
        self.dt = dt
        self.ts_per_frame = ts_per_frame
        self.nt = len(self.frames)

        # Ensure that vertices are reshaped into a grid format (if they are structured)
        # You may need to adjust this depending on the exact layout of your vertices
        self.X = self.vertices[:, 0].reshape((int(np.sqrt(len(vertices))), -1))  # Reshape X coordinates
        self.Y = self.vertices[:, 1].reshape((int(np.sqrt(len(vertices))), -1))  # Reshape Y coordinates
        self.Z = self.vertices[:, 2].reshape((int(np.sqrt(len(vertices))), -1))  # Reshape Z coordinates

        self.fig = plt.figure()

    def plotheatmap(self, k):
        plt.clf()

        ax = self.fig.add_subplot(111, projection='3d')
        ax.set_title(f"Temperature at t = {k * self.dt * self.ts_per_frame:.3f} unit time")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        u_k = self.frames[k].reshape(self.X.shape)
        
        surf = ax.plot_surface(self.X, 
                               self.Y, 
                               self.Z,
                               facecolors=plt.cm.jet((u_k - np.min(u_k)) / (np.max(u_k) - np.min(u_k))),  # Normalize `u_k`
                               cmap='jet',
                               rstride=1,
                               cstride=1,
                               linewidth=0,
                               antialiased=True,
                               vmin=np.min(u_k),  # Set the minimum value for the colormap
                               vmax=np.max(u_k)
                            )
        self.fig.colorbar(surf)

    def animate(self, k):
        self.plotheatmap(k)

    def save_gif(self, filepath):
        anim = animation.FuncAnimation(self.fig, self.animate, interval=1, frames=self.nt, repeat=False)
        anim.save(filepath)  


class Visualization3D:
    def __init__(self, vertices, tetrahedrons):
        self.vertices = vertices
        self.tetrahedrons = tetrahedrons
        self.n_tetrahedrons = tetrahedrons.shape[0]

    def save_frame(self, color_values, frame_path):
        cells = np.hstack((np.full((self.n_tetrahedrons, 1), 4, dtype=int), self.tetrahedrons)).flatten().astype(int)
        cell_type = np.full(self.n_tetrahedrons, 10, dtype=int)
        grid = pv.UnstructuredGrid(cells, cell_type, self.vertices)
        grid.point_data["colors"] = color_values

        # Plot the mesh
        # plotter = pv.Plotter()
        # plotter.add_mesh(grid, scalars="colors", cmap="viridis", show_edges=True)
        # plotter.show()
        file_path = Path(frame_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        grid.save(frame_path)



