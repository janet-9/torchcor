import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch


class Visualization:
    def __init__(self, frames, triangulation, dt, ts_per_frame):
        self.frames = frames.cpu().numpy()
        self.triangulation = triangulation
        self.dt = dt
        self.ts_per_frame = ts_per_frame
        self.nt = len(frames)

    def plotheatmap(self, tr, u_k, k):
        plt.clf()

        plt.title(f"Temperature at t = {k * self.dt * self.ts_per_frame:.3f} unit time")
        plt.xlabel("x")
        plt.ylabel("y")

        plt.tripcolor(tr, u_k.flatten(), cmap=plt.cm.jet, vmin=0, vmax=100, shading='gouraud')
        plt.colorbar()

    def animate(self, k):
        self.plotheatmap(self.triangulation, self.frames[k], k)

    def save_gif(self, filepath):
        anim = animation.FuncAnimation(plt.figure(), self.animate, interval=1, frames=self.nt, repeat=False)
        anim.save(filepath)


