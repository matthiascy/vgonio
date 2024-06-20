import argparse

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from ..utils import uniform_disk_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--surf", action="store_true", help="Plot the surface")
    parser.add_argument("--gen", action="store_true", help="Generate figures")
    args = parser.parse_args()

    sns.set(style="whitegrid", color_codes=True)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    pnt_size = 4
    xs, ys = uniform_disk_samples(2048)

    dtheta = np.pi / 8
    for i, ax in enumerate(axs):
        theta = i * dtheta
        ax.set_title(fr'$\theta_i={np.degrees(theta):.1f}\degree$', fontsize=24)
        ax.set_aspect('equal')
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])
        ax.grid(True, linestyle='--', linewidth=0.5)  # Subtle gridlines
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        factor = np.cos(theta)
        if args.surf:
            x = np.cos(np.pi / 4) * 0.95
            y = np.sin(np.pi / 4) * factor * 0.95
            rectangle = patches.Rectangle((-x, -y), 2.0 * x, 2.0 * y, linewidth=0, facecolor='seagreen', alpha=0.15)
            ax.add_patch(rectangle)
        ax.scatter(xs, ys * factor, s=pnt_size, c='b', linewidth=0.1, alpha=0.6)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.0)

    if args.gen:
        plt.savefig("light_samples.pdf")
    else:
        plt.show()
