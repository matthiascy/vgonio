"""
The normal distribution function (microfacet distribution)
"""
import os.path
import sys
from io import StringIO

import argparse
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import re


def read_data(filename):
    """
    Read the data from the given file.

    Parameters
    ----------
    filename : str
        The name of the file to read the data from.

    Returns
    -------
    ndarray with dimension 2
        The data read from the file.
    azimuth bin size : float (degrees)
    zenith bin size : float (degrees)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
        # first group is the bin size, second group is the bin count
        pattern = re.compile(r"^.*:\s([\d|.]+)°.*:\s(\d+)$")
        if lines[0] == "microfacet distribution\n":
            azimuth_result = re.match(pattern, lines[1]).groups()
            zenith_result = re.match(pattern, lines[2]).groups()
            return np.genfromtxt(StringIO(lines[3]), dtype=np.float32, delimiter=' ')\
                .reshape(int(azimuth_result[1]), int(zenith_result[1])), \
                float(azimuth_result[0]), float(zenith_result[0])
        else:
            raise ValueError("The file does not contain the correct data: microfacet distribution required.")


def convert_to_xyz(data, azimuth_bins, zenith_bins):
    """
    Convert the data from the polar coordinate system to the cartesian coordinate system.
    Z-axis is the distribution value.
    """
    azimuth_bin_count = len(azimuth_bins)
    zenith_bin_count = len(zenith_bins)
    count = azimuth_bin_count * zenith_bin_count
    xs = np.empty(count, dtype=np.float32)
    ys = np.empty(count, dtype=np.float32)
    zs = np.empty(count, dtype=np.float32)
    for i, azimuth in enumerate(azimuth_bins):
        for j, zenith in enumerate(zenith_bins):
            n = i * zenith_bin_count + j;
            theta = np.radians(zenith)
            phi = np.radians(azimuth)
            xs[n] = np.sin(theta) * np.cos(phi)
            ys[n] = np.sin(theta) * np.sin(phi)
            zs[n] = data[i, j]
    return xs, ys, zs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the microfacet distribution.")
    parser.add_argument("filename", help="The file to read the data from.")
    parser.add_argument("-t", "--in-3d", action="store_true", help="Plot the data in 3D.")
    parser.add_argument("-s", "--save", action="store_true", help="Save the plots to file.")
    parser.add_argument("-p", "--phi", nargs='*', type=float, help="The azimuth angles to plot, in degrees.")
    args = parser.parse_args()
    sb.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.5)
    data, azimuth_bin_size, zenith_bin_size = read_data(sys.argv[1])
    azimuth_bins = np.arange(0, 360, azimuth_bin_size)
    zenith_bins = np.arange(0, 90 + zenith_bin_size, zenith_bin_size)
    xs, ys, zs = convert_to_xyz(data, azimuth_bins, zenith_bins)

    basename = os.path.basename(args.filename)
    out_dir = f"{basename.split('.')[0]}_plots"

    figures = []
    if args.in_3d:
        figures.append((plt.figure(), "microfacet_distribution_3D.png"))
        ax = figures[-1][0].add_subplot(projection='3d')
        ax.set_title(f"Microfacet distribution of {basename.split('-')[-1]}")
        ax.plot_trisurf(xs, ys, zs, cmap='viridis', edgecolor='none')
    for phi in args.phi:
        figures.append((plt.figure(), f"microfacet_distribution_phi={phi:.2f}.png"))
        ax = figures[-1][0].add_subplot()
        ax.set_title(f"Microfacet distribution of {basename.split('-')[-1]}, azimuth angle {phi:.2f}°")
        phi_idx_right = int(phi / azimuth_bin_size)
        phi_idx_left = int(((phi + 180.0) % 360.0) / azimuth_bin_size)
        zenith_bins_full = np.arange(-90, 90 + zenith_bin_size, zenith_bin_size)
        ticks = np.arange(-90, 90 + zenith_bin_size, 15)
        ax.tick_params(axis='x', which='major', labelsize=10)
        ax.set_xticks(ticks, labels=map(lambda x: f"{x}°", ticks))
        ax.set_xlabel("polar angle")
        ax.set_ylabel("per steradian")
        ax.plot(zenith_bins_full,
                np.concatenate((data[phi_idx_left, :][::-1], data[phi_idx_right, 1:])))
    if not args.save:
        plt.show()
    else:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for fig, filename in figures:
            fig.savefig(os.path.join(out_dir, filename))
