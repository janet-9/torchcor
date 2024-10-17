import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Visualization:
    def __init__(self, U, triangulation, dt):
        self.U = U.cpu()
        self.triangulation = triangulation
        self.dt = dt
        self.nt = self.U.shape[0]

    def plotheatmap(self, tr, u_k, k):
        plt.clf()

        plt.title(f"Temperature at t = {k * self.dt:.3f} unit time")
        plt.xlabel("x")
        plt.ylabel("y")

        plt.tripcolor(tr, u_k.flatten(), cmap=plt.cm.jet, vmin=0, vmax=100, shading='gouraud')
        plt.colorbar()

    def animate(self, k):
        self.plotheatmap(self.triangulation, self.U[k], k)

    def save_gif(self, filepath):
        anim = animation.FuncAnimation(plt.figure(), self.animate, interval=1, frames=self.nt, repeat=False)
        anim.save(filepath)


