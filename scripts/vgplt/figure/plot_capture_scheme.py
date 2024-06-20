import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse

from ..utils import points_on_hemisphere
from ..hemisphere import hemi_coord_figure


def plot_points(ax, r=1.0, c='m', s=15, extra=False):
    """Plot a set of points on the hemisphere."""
    x, y, z, theta, phi = points_on_hemisphere(r)
    ax.scatter(x, y, z, color=c, s=s)
    if extra:
        ax.scatter(r * np.sin(theta[0:, 0]) * np.cos(phi[0:, 0]),
                   r * np.sin(theta[0:, 0]) * np.sin(phi[0:, 0]),
                   r * np.cos(theta[0:, 0]),
                   color='r', s=850, marker='o', alpha=0.4)


def plot_patches(ax):
    from plot_beckers import becker_plot_hemisphere, compute_becker
    ks, rs, ts = compute_becker(10, True)
    becker_plot_hemisphere(ks, ts, ax)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnt", action="store_true", help="Plot points")
    parser.add_argument("--pch", action="store_true", help="Plot patches")
    parser.add_argument("--gen", action="store_true", help="Generate figures")
    parser.add_argument("--extra", action="store_true", help="Extra points")

    args = parser.parse_args()

    sns.set_theme(style="whitegrid", color_codes=True)

    if args.pnt:
        fig, ax = hemi_coord_figure(axes='')
        plot_points(ax, extra=args.extra)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
        if args.gen:
            plt.savefig("hemi-points.pdf")

    if args.pch:
        fig, ax = hemi_coord_figure(axes='')
        plot_patches(ax)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
        if args.gen:
            plt.savefig("hemi-patches.pdf")

    if (args.pch or args.pnt) and not args.gen:
        plt.show()
