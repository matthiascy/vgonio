"""
The normal distribution function (microfacet distribution)
"""
import io
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
        header = f.read(48)
        if header[0:4] != b'VGMO' or header[4] != ord(b'\x01'):
            raise Exception('Invalid file format, the file does not contain the correct data: microfacet distribution required.')
        is_binary = header[5] == ord('!')
        is_compressed = header[6] != 0

        print(f"is_binary = {is_binary}")
        print(f"is_compressed = {is_compressed}")

        [azimuth_start, azimuth_stop, azimuth_bin_size] = np.degrees(struct.unpack("<fff", header[8:20]))
        (azimuth_bin_count,) = struct.unpack("<I", header[20:24])
        print(f"azimuth -- start = {azimuth_start}, stop = {azimuth_stop}, bin_size = {azimuth_bin_size}, count = {azimuth_bin_count}")

        [zenith_start, zenith_stop, zenith_bin_size] = np.degrees(struct.unpack("<fff", header[24:36]))
        (zenith_bin_count,) = struct.unpack("<I", header[36:40])
        print(f"zenith -- start = {zenith_start}, stop = {zenith_stop}, bin_size = {zenith_bin_size}, count = {zenith_bin_count}")

        (sample_count,) = struct.unpack("<I", header[40:44])
        print(f"sample_count = {sample_count}")

        if sample_count != azimuth_bin_count * zenith_bin_count:
            raise Exception('Invalid file format, {} samples expected, but {} samples found.'.format(azimuth_bin_count * zenith_bin_count, sample_count))

        print('azimuth_start = {}, azimuth_stop = {}, azimuth_bin_size = {}, azimuth_bin_count = {}'
              .format(azimuth_start, azimuth_stop, azimuth_bin_size, azimuth_bin_count))
        print('zenith_start = {}, zenith_stop = {}, zenith_bin_size = {}, zenith_bin_count = {}'
              .format(zenith_start, zenith_stop, zenith_bin_size, zenith_bin_count))

        if is_compressed:
            if header[6] == 1:
                import zlib as compression
            else:
                import gzip as compression
            print('decompressing data')
            f = io.BytesIO(compression.decompress(f.read()))
            if is_binary:
                print('read compressed binary file')
                data = np.frombuffer(f.read(), dtype=('<f'), count=sample_count)
            else:
                print('read compressed ascii file')
                data = np.loadtxt(f, dtype=np.float32, delimiter=' ')
        else:
            if is_binary:
                print('read uncompressed binary file')
                data = np.fromfile(f, dtype=('<f'), count=sample_count)
            else:
                print('read uncompressed ascii file')
                data = np.fromfile(f, dtype=np.float32, count=sample_count, sep=' ')

        data = data.reshape((azimuth_bin_count, zenith_bin_count))

        return data, azimuth_start, azimuth_stop, azimuth_bin_size, azimuth_bin_count, zenith_start, zenith_stop, zenith_bin_size, zenith_bin_count


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
            n = i * zenith_bin_count + j
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
    data, azimuth_start, azimuth_stop, azimuth_bin_size, azimuth_bin_count, zenith_start, zenith_stop, zenith_bin_size, zenith_bin_count = read_data(args.filename)
    azimuth_bins = np.zeros(azimuth_bin_count, dtype=np.float32)
    zenith_bins = np.zeros(zenith_bin_count, dtype=np.float32)
    for i in range(azimuth_bin_count):
        azimuth_bins[i] = azimuth_start + i * azimuth_bin_size
    for i in range(zenith_bin_count):
        zenith_bins[i] = zenith_start + i * zenith_bin_size
    if len(azimuth_bins) != data.shape[0] or len(zenith_bins) != data.shape[1]:
        raise Exception('The data step size does not match the data shape.')

    xs, ys, zs = convert_to_xyz(data, azimuth_bins, zenith_bins)

    basename = os.path.basename(args.filename)
    output_dir = f"{basename.split('.')[0]}_plots"

    micro_surface_name = basename.split('-')[-1].split('.')[0]

    figures = []
    if args.in_3d:
        figures.append((plt.figure(), "microfacet_normal_distribution_3d.png"))
        ax = figures[-1][0].add_subplot(projection='3d')
        ax.set_title(f"Microfacet NDF - {micro_surface_name}")
        ax.plot_trisurf(xs, ys, zs, cmap='viridis', edgecolor='none')
    for phi in args.phi:
        print(f"Plotting {phi}°")
        figures.append((plt.figure(), f"mndf_phi={phi:.2f}.png"))
        ax = figures[-1][0].add_subplot()
        ax.set_title(f"Microfacet NDF - {micro_surface_name}")
        phi_idx_right = int((phi - azimuth_start) / azimuth_bin_size)
        phi_idx_left = int(((phi - azimuth_start + 180.0) % 360.0) / azimuth_bin_size)
        print(f"phi_idx_left={phi_idx_left}, phi_idx_right={phi_idx_right}")
        zenith_bins_full_range = np.concatenate([np.flip(zenith_bins[1:]) * -1.0, zenith_bins])
        ticks = np.arange(-90, 90 + zenith_bin_size, 15)
        ax.tick_params(axis='x', which='major', labelsize=10)
        ax.set_xticks(ticks, labels=map(lambda x: f"{x:.0f}°", ticks))
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$sr^{-1}$")
        ax.plot(zenith_bins_full_range,
                np.concatenate((data[phi_idx_left, :][::-1], data[phi_idx_right, 1:])))
        phi_right = azimuth_bins[phi_idx_right]
        phi_left = azimuth_bins[phi_idx_left]
        ax.annotate(r"$\phi = {:.2f}^\circ$".format(phi_left), xy=(0.05, 0.95), xycoords='axes fraction', xytext=(0.05, 0.3),
                    bbox=dict(boxstyle="round", fc="none", ec="gray"))
        ax.annotate(r"$\phi = {:.2f}^\circ$".format(phi_right), xy=(0.0, 0.0), xycoords='axes fraction', xytext=(0.75, 0.3),
                    bbox=dict(boxstyle="round", fc="none", ec="gray"))

    if not args.save:
        plt.show()
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for fig, filename in figures:
            fig.savefig(os.path.join(output_dir, filename))
