import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

class Visualization:
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

    def plotheatmap(self, k):
        plt.clf()

        fig = plt.figure()  # Create a new figure for each frame
        ax = fig.add_subplot(111, projection='3d')

        ax.clear()  # Clear the previous frame
        ax.set_title(f"Temperature at t = {k * self.dt * self.ts_per_frame:.3f} unit time")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        u_k = self.frames[k]

        # Plot the surface
        surf = ax.plot_trisurf(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2],
                        triangles=self.triangles, cmap='jet', facecolors=plt.cm.jet(u_k.flatten() / np.max(u_k)),
                        linewidth=0)
        fig.colorbar(surf)
        plt.show()

    def animate(self, k):
        self.plotheatmap(k)

    def save_gif(self, filepath):
        fig = plt.figure()
        anim = animation.FuncAnimation(fig, self.animate, interval=1, frames=self.nt, repeat=False)
        anim.save(filepath, writer='pillow')  # Change to 'pillow' if ImageMagick is not available



