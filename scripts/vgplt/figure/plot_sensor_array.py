import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

from ..utils import points_on_hemisphere, rotate, path_patch_2d_to_3d
from ..hemisphere import hemi_coord_figure


def plot_circle(ax, theta, phi, r=0.1, **kwargs):
    m = np.matmul(rotate(phi, 'z'), rotate(theta, 'y'))
    circle = Circle((0, 0), r, **kwargs)
    ax.add_patch(circle)
    path_patch_2d_to_3d(circle, m, z=1)


def plot_figure(r, alpha):
    fig, ax = hemi_coord_figure(surf=True, axes_alpha=(0.5, 0.5, 0.5), ha=0.2, hc='m')
    xs, ys, zs, ts, ps = points_on_hemisphere(ntheta=10, nphi=15)
    for i in range(2):
        ax.scatter(xs[:, i], ys[:, i], zs[:, i], color='m', s=10, marker='o', alpha=0.8)
    for i in range(2, 15):
        ax.scatter(xs[:, i], ys[:, i], zs[:, i], color='m', s=10, marker='o', alpha=0.2)

    for p in ps[0, :][:2]:
        for t in ts[:-1, 0]:
            plot_circle(ax, t, p, r=r, color='indianred', alpha=alpha, linewidth=0, fill=True)

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", action="store_true", help="Generate figures")
    parser.add_argument("--large", action="store_true", help="Large sensors")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", color_codes=True)

    if args.gen:
        plot_figure(0.08, 0.4)
        plt.savefig("hemi-sensors-small.pdf")
        plot_figure(0.15, 0.2)
        plt.savefig("hemi-sensors-large.pdf")
    else:
        r = 0.08 if not args.large else 0.15
        alpha = 0.4 if not args.large else 0.28
        plot_figure(r, alpha)
        plt.show()
