"""
The shadowing-masking term G depends on the distribution function D
and the details of the micro-surface.
"""
import argparse
import math
import struct
import os
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import io
import zlib

def read_data(filename):
    with open(filename, 'rb') as f:
        header = f.read(48)
        if header[0:4] != b'VGMO' or header[4] != ord(b'\x02'):
            raise Exception('Invalid file format, the file does not contain the correct data: microfacet masking shadowing required.')
        is_binary = header[5] == ord('!')
        is_compressed = header[6] != 0
        [azimuth_start, azimuth_stop, azimuth_bin_size] = np.degrees(struct.unpack("<fff", header[8:20]))
        azimuth_bin_count = int.from_bytes(header[20:24], byteorder='little')
        [zenith_start, zenith_stop, zenith_bin_size] = np.degrees(struct.unpack("<fff", header[24:36]))
        zenith_bin_count = int.from_bytes(header[36:40], byteorder='little')
        sample_count = int.from_bytes(header[40:44], byteorder='little')

        if sample_count != (azimuth_bin_count * zenith_bin_count) ** 2:
            raise Exception('Invalid file format, sample count does not match the number of bins.')

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
            n = i * zenith_bin_count + j
            theta = math.radians(zenith)
            phi = math.radians(azimuth)
            xs[n] = round(np.sin(theta) * np.cos(phi), 8)
            ys[n] = round(np.sin(theta) * np.sin(phi), 8)
            zs[n] = data[azimuth_m_idx, zenith_m_idx, i, j]
    return xs, ys, zs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microfacet shadowing/masking term plotting")
    parser.add_argument("-f", "--filename", help="The file to read the data from.")
    parser.add_argument("-mt", "--m-theta", nargs="*", type=float, help="The zenith angle (theta) of the micro-facet normal (m).")
    parser.add_argument("-mp", "--m-phi", nargs="*", type=float, help="The azimuthal angle (phi) of the micro-facet normal (m).")
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

    print('azimuth bins: ', azimuth_bins)
    print('zenith bins: ', zenith_bins)

    m_azimuths = args.m_phi
    m_zeniths = args.m_theta
    m_azimuth_indices = list(map(lambda azimuth: int(round((azimuth - azimuth_start) / azimuth_bin_size)), m_azimuths))
    m_zenith_indices = list(map(lambda zenith: int(round((zenith - zenith_start) / zenith_bin_size)), m_zeniths))

    print('m azimuths indices: ', m_azimuth_indices)
    print('m zeniths indices: ', m_zenith_indices)

    # MSF data classified by m
    data_by_m = []
    for m_azimuth_idx in m_azimuth_indices:
        for m_zenith_idx in m_zenith_indices:
            data_by_m.append((
                m_azimuth_idx, m_zenith_idx,
                convert_to_xyz(data, m_azimuth_idx, m_zenith_idx, azimuth_bins, zenith_bins)))

    basename = os.path.basename(args.filename)
    output_dir = f"{basename.split('.')[0]}_plots"
    micro_surface_name = basename.split('-')[-1].split('.')[0]

    figures = []
    for m_azimuth_idx, m_zenith_idx, (xs, ys, zs) in data_by_m:
        m_zenith = zenith_bins[m_zenith_idx]
        m_azimuth = azimuth_bins[m_azimuth_idx]
        if args.in_3d:
            figures.append((plt.figure(), f"microfacet_masking_shadowing_m({m_zenith},{m_azimuth})_3d-png"))
            ax = figures[-1][0].add_subplot(projection='3d')
            ax.set_title(f"Microfacet shadowing/masking function - {micro_surface_name}")
            ax.scatter(xs, ys, zs, cmap='viridis', edgecolor='none')
            ax.annotate(r"$m = (\theta={:.2f}, \phi={:.2f})$".format(m_zenith, m_azimuth), xy=(0.05, 1.0),
                        xycoords='axes fraction', xytext=(-0.32, -0.1),
                        bbox=dict(boxstyle="round", fc="none", ec="gray"))
            ax.set_zlim(0.0, 1.0)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_zlabel(r"$z$")
        elif args.phi is not None:
            for phi in args.phi:
                m_zenith = zenith_bins[m_zenith_idx]
                m_azimuth = azimuth_bins[m_azimuth_idx]
                figures.append((plt.figure(), f"mmsf_m({m_zenith:.2f},{m_azimuth:.2f})_phi={phi:.2f}.png"))
                ax = figures[-1][0].add_subplot()
                ax.set_title(r"$G_1(\mathbf{{i}},\mathbf{{m}})$ - {}".format(micro_surface_name))
                ax.set_xlabel(r"$\theta$")
                ax.set_ylabel(r"ratio")
                phi_idx_right = int(np.ceil((phi - azimuth_start) / azimuth_bin_size))
                phi_idx_left = int(np.ceil(((phi - azimuth_start + 180.0) % 360.0) / azimuth_bin_size))
                zenith_bins_full_range = np.concatenate([np.flip(zenith_bins[1:]) * -1.0, zenith_bins])
                ticks = np.arange(-90, 90 + zenith_bin_size, 15)
                ax.tick_params(axis='x', which='major', labelsize=10)
                ax.set_xticks(ticks, labels=[f"{np.abs(t):.0f}°" for t in ticks])
                ax.plot(zenith_bins_full_range, np.concatenate(
                    (
                        data[m_azimuth_idx, m_zenith_idx, phi_idx_left, :][::-1],
                        data[m_azimuth_idx, m_zenith_idx, phi_idx_right, 1:]
                    )))
                phi_right = azimuth_bins[phi_idx_right]
                phi_left = azimuth_bins[phi_idx_left]
                print(f"phi_right {phi_idx_right}: {phi_right:.2f}, phi_left {phi_idx_left}: {phi_left:.2f}")
                ax.annotate(r"$\mathbf{{i}}_{{\phi}} = {:.2f}^\circ$".format(phi_left), xy=(0.05, 0.95), xycoords='axes fraction', xytext=(0.05, 0.3),
                            bbox=dict(boxstyle="round", fc="none", ec="gray"))
                ax.annotate(r"$\mathbf{{i}}_{{\phi}} = {:.2f}^\circ$".format(phi_right), xy=(0.0, 0.0), xycoords='axes fraction', xytext=(0.75, 0.3),
                            bbox=dict(boxstyle="round", fc="none", ec="gray"))
                ax.annotate(r"$\mathbf{{m}}_{{(\theta, \phi)}} = ({:.2f}^\circ, {:.2f}^\circ)$".format(m_zenith, m_azimuth),
                            xy=(0.5, 0.5),
                            xycoords='axes fraction',
                            xytext=(0.27, 0.05),
                            bbox=dict(boxstyle="round", fc="none", ec="gray"))

    if len(figures) > 0:
        if not args.save:
            plt.show()
        else:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for fig, name in figures:
                fig.savefig(os.path.join(output_dir, name))
