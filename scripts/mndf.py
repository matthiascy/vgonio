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


class MicrofacetDistribution:
    def __init__(self, samples, azimuth_start, azimuth_stop, azimuth_bin_size, azimuth_bin_count, zenith_start,
                 zenith_stop, zenith_bin_size, zenith_bin_count):
        self.samples = samples
        self.azimuth_start = azimuth_start
        self.azimuth_stop = azimuth_stop
        self.azimuth_bin_size = azimuth_bin_size
        self.azimuth_bin_count = azimuth_bin_count
        self.zenith_start = zenith_start
        self.zenith_stop = zenith_stop
        self.zenith_bin_size = zenith_bin_size
        self.zenith_bin_count = zenith_bin_count


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
        metadata = f.read(49)
        print(metadata[0:4])
        if metadata[0:4] != b'VGMO':
            raise Exception(
                'Invalid file format, the file does not contain the correct data: microfacet distribution required.')
        (major, minor, patch, _) = struct.unpack('<BBBB', metadata[4:8])
        (file_size,) = struct.unpack('<I', metadata[8:12])
        timestamp = metadata[12:44]
        (sample_size,) = struct.unpack('<B', metadata[44:45])
        is_binary = metadata[45:46] == b'!'
        (_compression_type,) = struct.unpack('<B', metadata[46:47])
        is_compressed = _compression_type != 0
        compression_type = 'none' if not is_compressed else 'zlib' if _compression_type == 1 else 'gzip' if _compression_type == 2 else 'unknown'
        (mtype,) = struct.unpack('<B', metadata[48:49])
        measurement_type = 'BRDF' if mtype == 0 else 'NDF' if mtype == 1 else 'MSF' if mtype == 2 else 'unknown'

        print(f"is_binary = {is_binary}")
        print(f"is_compressed = {is_compressed}")
        print(f"major = {major}, minor = {minor}, patch = {patch}")
        print(f"file_size = {file_size}")
        print(f"timestamp = {timestamp}")
        print(f"sample_size = {sample_size}")
        print(f"measurement_type = {measurement_type}")
        print(f"compression_type = {compression_type}")

        if measurement_type != 'NDF':
            raise Exception('Invalid file format, microfacet distribution required.')

        ndf_header = f.read(36)

        [azimuth_start, azimuth_stop, azimuth_bin_size] = np.degrees(struct.unpack("<fff", ndf_header[0:12]))
        (azimuth_bin_count,) = struct.unpack("<I", ndf_header[12:16])
        print(
            f"azimuth -- start = {azimuth_start}, stop = {azimuth_stop}, bin_size = {azimuth_bin_size}, count = {azimuth_bin_count}")

        [zenith_start, zenith_stop, zenith_bin_size] = np.degrees(struct.unpack("<fff", ndf_header[16:28]))
        (zenith_bin_count,) = struct.unpack("<I", ndf_header[28:32])
        print(
            f"zenith -- start = {zenith_start}, stop = {zenith_stop}, bin_size = {zenith_bin_size}, count = {zenith_bin_count}")

        (sample_count,) = struct.unpack("<I", ndf_header[32:36])
        print(f"sample_count = {sample_count}")

        if sample_count != azimuth_bin_count * zenith_bin_count:
            raise Exception('Invalid file format, {} samples expected, but {} samples found.'.format(
                azimuth_bin_count * zenith_bin_count, sample_count))

        print('azimuth_start = {}, azimuth_stop = {}, azimuth_bin_size = {}, azimuth_bin_count = {}'
              .format(azimuth_start, azimuth_stop, azimuth_bin_size, azimuth_bin_count))
        print('zenith_start = {}, zenith_stop = {}, zenith_bin_size = {}, zenith_bin_count = {}'
              .format(zenith_start, zenith_stop, zenith_bin_size, zenith_bin_count))

        if is_compressed:
            if compression_type == 'zlib':
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

        return MicrofacetDistribution(data, azimuth_start, azimuth_stop, azimuth_bin_size, azimuth_bin_count,
                                      zenith_start, zenith_stop, zenith_bin_size, zenith_bin_count)
        # return data, azimuth_start, azimuth_stop, azimuth_bin_size, azimuth_bin_count, zenith_start, zenith_stop, zenith_bin_size, zenith_bin_count


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
    parser.add_argument("filename", nargs='*', help="The file to read the data from.")
    parser.add_argument("-t", "--in-3d", action="store_true", help="Plot the data in 3D.")
    parser.add_argument("-s", "--save", action="store_true", help="Save the plots to file.")
    parser.add_argument("-p", "--phi", nargs='*', type=float, help="The azimuth angles to plot, in degrees.")
    args = parser.parse_args()
    sb.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.5)
    ndfs = []
    for filename in args.filename:
        ndfs.append(read_data(filename))

    # Here we assume that the azimuth and zenith bins are the same for all NDFs
    azimuth_bins = np.zeros(ndfs[0].azimuth_bin_count, dtype=np.float32)
    zenith_bins = np.zeros(ndfs[0].zenith_bin_count, dtype=np.float32)
    for i in range(ndfs[0].azimuth_bin_count):
        azimuth_bins[i] = ndfs[0].azimuth_start + i * ndfs[0].azimuth_bin_size
    for i in range(ndfs[0].zenith_bin_count):
        zenith_bins[i] = ndfs[0].zenith_start + i * ndfs[0].zenith_bin_size
    if len(azimuth_bins) != ndfs[0].samples.shape[0] or len(zenith_bins) != ndfs[0].samples.shape[1]:
        raise Exception('The data step size does not match the data shape.')

    xss, yss, zss = [], [], []
    for ndf in ndfs:
        xs, ys, zs = convert_to_xyz(ndf.samples, azimuth_bins, zenith_bins)
        xss.append(xs)
        yss.append(ys)
        zss.append(zs)

    output_dir = []
    micro_surface_name = []
    for filename in args.filename:
        basename = os.path.basename(filename)
        output_dir.append(f"{basename.split('.')[0]}_plots")
        micro_surface_name.append(basename.split('-')[-1].split('.')[0])

    figures = []
    if args.in_3d:
        for xs, ys, zs in zip(xss, yss, zss):
            figures.append((plt.figure(), "microfacet_normal_distribution_3d.png"))
            ax = figures[-1][0].add_subplot(projection='3d')
            ax.set_title(f"Microfacet NDF - {micro_surface_name}")
            ax.plot_trisurf(xs, ys, zs, cmap='viridis', edgecolor='none')
    for phi in args.phi:
        phi_idx_right = int((phi - ndfs[0].azimuth_start) / ndfs[0].azimuth_bin_size)
        phi_idx_left = int(((phi - ndfs[0].azimuth_start + 180.0) % 360.0) / ndfs[0].azimuth_bin_size)
        phi_right = azimuth_bins[phi_idx_right]
        phi_left = azimuth_bins[(phi_idx_left + 1) % len(azimuth_bins)]

        figures.append((plt.figure(), f"mndf_all_phi={phi:.2f}.png"))
        ax_all = figures[-1][0].add_subplot()
        ax_all.set_xlabel(r"$\theta$")
        ax_all.set_ylabel(r"$sr^{-1}$")
        ax_all.set_title(f"Microfacet NDF of subdivided surface")
        ax_all.annotate(r"$\phi = {:.2f}^\circ$".format(phi_left), xy=(0.05, 0.95), xycoords='axes fraction',
                        xytext=(0.05, 0.3),
                        bbox=dict(boxstyle="round", fc="none", ec="gray"))
        ax_all.annotate(r"$\phi = {:.2f}^\circ$".format(phi_right), xy=(0.0, 0.0), xycoords='axes fraction',
                        xytext=(0.75, 0.3),
                        bbox=dict(boxstyle="round", fc="none", ec="gray"))
        for ndf, name in zip(ndfs, micro_surface_name):
            print(f"Plotting {phi}°")
            figures.append((plt.figure(), f"mndf_phi={phi:.2f}.png"))
            ax = figures[-1][0].add_subplot()
            ax.set_title(f"Microfacet NDF - {name}")
            print(f"phi_idx_left={phi_idx_left}, phi_idx_right={phi_idx_right}")
            zenith_bins_full_range = np.concatenate([np.flip(zenith_bins[1:]) * -1.0, zenith_bins])
            ticks = np.arange(-90, 90 + ndf.zenith_bin_size, 15)
            ax.tick_params(axis='x', which='major', labelsize=10)
            ax.set_xticks(ticks, labels=map(lambda x: f"{x:.0f}°", ticks))
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$sr^{-1}$")
            ax.plot(zenith_bins_full_range,
                    np.concatenate((ndf.samples[phi_idx_left, :][::-1], ndf.samples[phi_idx_right, 1:])))
            ax.annotate(r"$\phi = {:.2f}^\circ$".format(phi_left), xy=(0.05, 0.95), xycoords='axes fraction',
                        xytext=(0.05, 0.3),
                        bbox=dict(boxstyle="round", fc="none", ec="gray"))
            ax.annotate(r"$\phi = {:.2f}^\circ$".format(phi_right), xy=(0.0, 0.0), xycoords='axes fraction',
                        xytext=(0.75, 0.3),
                        bbox=dict(boxstyle="round", fc="none", ec="gray"))
            ax_all.plot(zenith_bins_full_range,
                        np.concatenate((ndf.samples[phi_idx_left, :][::-1], ndf.samples[phi_idx_right, 1:])),
                        label=name)
        ax_all.legend()

    if not args.save:
        plt.show()
    else:
        for output_dir in output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for fig, filename in figures:
                fig.savefig(os.path.join(output_dir, filename))
