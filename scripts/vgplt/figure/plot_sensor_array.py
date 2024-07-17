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


def plot_figure(r, alpha, elev=45, azim=-30, axes='xyz', hc='m', pc='m', cc='indianred', pn=2, same_alpha=False,
                all_theta=False):
    fig, ax = hemi_coord_figure(surf=True, axes_alpha=(0.5, 0.5, 0.5), ha=0.2, hc=hc, axes=axes, elev=elev, azim=azim)
    xs, ys, zs, ts, ps = points_on_hemisphere(ntheta=10, nphi=15)
    a0, a1 = (0.8, 0.8) if same_alpha else (0.8, 0.2)
    for i in range(2):
        ax.scatter(xs[:, i], ys[:, i], zs[:, i], color=pc, s=10, marker='o', alpha=a0)
    for i in range(2, 15):
        ax.scatter(xs[:, i], ys[:, i], zs[:, i], color=pc, s=10, marker='o', alpha=a1)

    for p in ps[0, :][:pn]:
        if all_theta:
            for t in ts[:, 0]:
                plot_circle(ax, t, p, r=r, color=cc, alpha=alpha, linewidth=0, fill=True)
        else:
            for t in ts[:-1, 0]:
                plot_circle(ax, t, p, r=r, color=cc, alpha=alpha, linewidth=0, fill=True)

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", action="store_true", help="Generate figures")
    parser.add_argument("--large", action="store_true", help="Large sensors")
    parser.add_argument("--hc", type=str, default='m', help="Hemisphere colour")
    parser.add_argument("--r", type=float, default=0.08, help="Radius of the circle")
    parser.add_argument("--ndf-sensors", action="store_true", help="Number of sensors in the NDF")
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", color_codes=True)

    if args.gen:
        plot_figure(0.08, 0.4)
        plt.savefig("hemi-sensors-small.pdf")
        plot_figure(0.15, 0.2)
        plt.savefig("hemi-sensors-large.pdf")
    else:
        r = args.r
        r = 0.088 if args.ndf_sensors else r
        alpha = 0.4 if not args.large else 0.28
        pn = 1 if args.ndf_sensors else 2
        same_alpha = args.ndf_sensors
        all_theta = args.ndf_sensors
        plot_figure(r, alpha, hc=args.hc, pn=pn, same_alpha=same_alpha, all_theta=all_theta, axes='', azim=35)
        plt.show()
