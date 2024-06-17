import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse


def setup_hemisphere_figure(with_surface=True):
    # Set colours and render
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([0.0, 1.0])
    ax.set_aspect("equal")
    ax.set_proj_type('ortho')

    r = 1
    # Create a hemisphere
    theta, phi = np.mgrid[0.0:np.pi / 2:100j, 0.0:2.0 * np.pi:100j]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='g', alpha=0.3, linewidth=0)
    ax.view_init(elev=45, azim=30)

    if with_surface:
        x = np.outer(np.linspace(-0.28, 0.28, 20), np.ones(20))
        y = x.copy().T
        z = (np.sin(x ** 2) + np.cos(y ** 2)) / 4 - 0.25
        ax.plot_surface(x, y, z, color='b', alpha=0.3, linewidth=0)

    # hide gridlines
    ax.grid(False)
    # hide y and z plane
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # hide x and z plane
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # hide axis line
    ax.xaxis.line.set_color("white")
    ax.yaxis.line.set_color("white")
    ax.zaxis.line.set_color("white")
    # hide ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    return fig, ax


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
    from beckers import becker_plot_hemisphere, compute_becker
    ks, rs, ts = compute_becker(1, 10)
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
        fig, ax = setup_hemisphere_figure()
        plot_points(ax, args.extra)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
        if args.gen:
            plt.savefig("hemi-points.pdf")

    if args.pch:
        fig, ax = setup_hemisphere_figure()
        plot_patches(ax)
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
        if args.gen:
            plt.savefig("hemi-patches.pdf")

    if (args.pch or args.pnt) and not args.gen:
        plt.show()
