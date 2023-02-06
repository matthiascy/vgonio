"""
The shadowing-masking term G depends on the distribution function D
and the details of the micro-surface.
"""
import argparse
import struct
import os
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

def read_data(filename):
    with open(filename, 'rb') as f:
        header = f.read(48)
        if header[0:4] != b'VGMO' or header[4] != ord(b'\x02'):
            raise Exception('Invalid file format, the file does not contain the correct data: microfacet masking shadowing required.')
        is_binary = header[5] == ord('!')
        [azimuth_start, azimuth_stop, azimuth_bin_size] = np.degrees(struct.unpack("fff", header[6:18]))
        azimuth_bin_count = int.from_bytes(header[18:22], byteorder='little')
        [zenith_start, zenith_stop, zenith_bin_size] = np.degrees(struct.unpack("fff", header[22:34]))
        zenith_bin_count = int.from_bytes(header[34:38], byteorder='little')
        print('azimuth_start = {}, azimuth_stop = {}, azimuth_bin_size = {}, azimuth_bin_count = {}'
              .format(azimuth_start, azimuth_stop, azimuth_bin_size, azimuth_bin_count))
        print('zenith_start = {}, zenith_stop = {}, zenith_bin_size = {}, zenith_bin_count = {}'
              .format(zenith_start, zenith_stop, zenith_bin_size, zenith_bin_count))
        sample_count = int.from_bytes(header[38:42], byteorder='little')
        if sample_count != (azimuth_bin_count * zenith_bin_count) ** 2:
            raise Exception('Invalid file format, sample count does not match the number of bins.')
        if is_binary:
            print('read binary file')
            data = np.fromfile(f, dtype=('<f'), count=sample_count)
        else:
            print('read text file')
            data = np.fromfile(f, dtype=np.float32, count=sample_count, sep=' ')

        data = data.reshape((azimuth_bin_count, zenith_bin_count, azimuth_bin_count, zenith_bin_count))

        return data, azimuth_start, azimuth_stop, azimuth_bin_size, azimuth_bin_count, zenith_start, zenith_stop, zenith_bin_size, zenith_bin_count


def convert_to_xyz(data, azimuth_m_idx, zenith_m_idx, azimuth_bins, zenith_bins):
    """
    Convert the data from the polar coordinate system to the cartesian coordinate system.
    Z-axis is the distribution value.
    """
    azimuth_bin_count = len(azimuth_bins)
    zenith_bin_count = len(zenith_bins)
    count = azimuth_bin_count * zenith_bin_count
    xs = np.zeros(count, dtype=np.float32)
    ys = np.zeros(count, dtype=np.float32)
    zs = np.zeros(count, dtype=np.float32)
    for i, azimuth in enumerate(azimuth_bins):
        for j, zenith in enumerate(zenith_bins):
            n = i * zenith_bin_count + j;
            theta = np.radians(zenith)
            phi = np.radians(azimuth)
            xs[n] = np.sin(theta) * np.cos(phi)
            ys[n] = np.sin(theta) * np.sin(phi)
            zs[n] = data[azimuth_m_idx, zenith_m_idx, i, j]
    return xs, ys, zs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microfacet shadowing/masking term plotting")
    parser.add_argument("filename", help="The file to read the data from.")
    parser.add_argument("-mt", "--m-theta", type=float, help="The zenith angle (theta) of the micro-facet normal (m).")
    parser.add_argument("-mp", "--m-phi", type=float, help="The azimuthal angle (phi) of the micro-facet normal (m).")
    parser.add_argument("-p", "--phi", nargs="*", type=float, help="The azimuthal angle to plot.")
    parser.add_argument("-t", "--in-3d", action="store_true", help="Plot the data in 3D.")
    parser.add_argument("-s", "--save", action="store_true", help="Save the plots to files.")
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

    azimuth_m = args.m_phi
    zenith_m = args.m_theta
    azimuth_m_idx = int((azimuth_m - azimuth_start) / azimuth_bin_size)
    zenith_m_idx = int((zenith_m - zenith_start) / zenith_bin_size)
    xs, ys, zs = convert_to_xyz(data, azimuth_m_idx, zenith_m_idx, azimuth_bins, zenith_bins)

    basename = os.path.basename(args.filename)
    output_dir = f"{basename.split('.')[0]}_plots"
    micro_surface_name = basename.split('-')[-1].split('.')[0]

    figures = []
    if args.in_3d:
        figures.append((plt.figure(), "microfacet_masking_shadowing_3d-png"))
        ax = figures[-1][0].add_subplot(projection='3d')
        ax.set_title(f"Microfacet shadowing/masking function - {micro_surface_name}")
        ax.scatter(xs, ys, zs, cmap='viridis', edgecolor='none')
        ax.annotate(r"$m = (\phi={:.2f}, \theta={:.2f})$".format(azimuth_m, zenith_m), xy=(0.05, 1.0),
                    xycoords='axes fraction', xytext=(-0.32, -0.1),
                    bbox=dict(boxstyle="round", fc="none", ec="gray"))
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")

    if not args.save:
        plt.show()
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for fig, name in figures:
            fig.savefig(os.path.join(output_dir, name))
