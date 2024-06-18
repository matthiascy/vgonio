import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse

from plot_hemisphere import new_hemisphere_figure


def plot_points(ax, extra=False):
    r = 1
    # generate a set of points with phi and theta
    theta, phi = np.mgrid[0.0:np.pi / 2:10j, 0.0:2.0 * np.pi:20j]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    ax.scatter(x, y, z, color='m', s=15)
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
        fig, ax = new_hemisphere_figure(with_axes=False)
        plot_points(ax, args.extra)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
        if args.gen:
            plt.savefig("hemi-points.pdf")

    if args.pch:
        fig, ax = new_hemisphere_figure(with_axes=False)
        plot_patches(ax)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
        if args.gen:
            plt.savefig("hemi-patches.pdf")

    if (args.pch or args.pnt) and not args.gen:
        plt.show()
