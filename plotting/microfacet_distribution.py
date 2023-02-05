"""
The normal distribution function (microfacet distribution)
"""
import os.path
import struct

import argparse
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


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
    with open(filename, 'rb') as f:
        header = f.read(40)
        if header[0:4] != b'VGMO' or header[4] != ord(b'\x01'):
            raise Exception('Invalid file format, the file does not contain the correct data: microfacet distribution required.')
        is_binary = header[5] == ord('!')
        [azimuth_start, azimuth_stop, azimuth_bin_size] = np.degrees(struct.unpack("fff", header[6:18]))
        azimuth_bin_count = int.from_bytes(header[18:22], byteorder='little')
        [zenith_start, zenith_stop, zenith_bin_size] = np.degrees(struct.unpack("fff", header[22:34]))
        zenith_bin_count = int.from_bytes(header[34:38], byteorder='little')
        print('azimuth_start = {}, azimuth_stop = {}, azimuth_bin_size = {}, azimuth_bin_count = {}'
              .format(azimuth_start, azimuth_stop, azimuth_bin_size, azimuth_bin_count))
        print('zenith_start = {}, zenith_stop = {}, zenith_bin_size = {}, zenith_bin_count = {}'
                .format(zenith_start, zenith_stop, zenith_bin_size, zenith_bin_count))
        if is_binary:
            print('read binary file')
            data = np.fromfile(f, dtype=('<f'), count=azimuth_bin_count * zenith_bin_count)
        else:
            print('read text file')
            data = np.fromfile(f, dtype=np.float32, count=azimuth_bin_count * zenith_bin_count, sep=' ')

        data = data.reshape((azimuth_bin_count, zenith_bin_count))

        return data, azimuth_start, azimuth_stop, azimuth_bin_size, zenith_start, zenith_stop, zenith_bin_size


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
    parser = argparse.ArgumentParser(description="Microfacet distribution plotting")
    parser.add_argument("filename", help="The file to read the data from.")
    parser.add_argument("-t", "--in-3d", action="store_true", help="Plot the data in 3D.")
    parser.add_argument("-s", "--save", action="store_true", help="Save the plots to file.")
    parser.add_argument("-p", "--phi", nargs='*', type=float, help="The azimuth angles to plot, in degrees.")
    args = parser.parse_args()
    sb.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.5)
    data, azimuth_start, azimuth_stop, azimuth_bin_size, zenith_start, zenith_stop, zenith_bin_size = read_data(args.filename)
    azimuth_bins = np.arange(azimuth_start, azimuth_stop, azimuth_bin_size)
    zenith_bins = np.arange(zenith_start, zenith_stop + zenith_bin_size, zenith_bin_size)
    xs, ys, zs = convert_to_xyz(data, azimuth_bins, zenith_bins)

    basename = os.path.basename(args.filename)
    output_dir = f"{basename.split('.')[0]}_plots"

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
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for fig, filename in figures:
            fig.savefig(os.path.join(output_dir, filename))
