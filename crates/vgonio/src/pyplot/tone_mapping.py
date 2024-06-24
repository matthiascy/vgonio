import argparse

import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_exr_channel(exr_file, channel):
    all_channels = exr_file.header()['channels'].keys()
    if channel not in all_channels:
        print(f"Channel '{channel}' not found in the EXR file. Available channels: {all_channels}")
        exit(1)

    # Read the luminance channel
    values_str = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
    values = np.frombuffer(values_str, dtype=np.float32)
    total_size = len(values)
    values = np.reshape(values, (size[1], size[0]))
    single_layer_size = size[0] * size[1]
    print(
        f"Loaded channel '{channel}' with shape {values.shape} and total size {total_size} = {total_size / single_layer_size} x {single_layer_size}")

    return values


def draw_polar_grid(ax, size, num_lines=8, num_circles=4):
    # Draw circles, each representing a certain angle
    # Do not draw the text of the angle if it is 0 or 90 degrees
    for r in np.linspace(0, size[1] / 2, num_circles):
        circle = plt.Circle((0, 0), r * 0.999, color='k', linestyle='-', linewidth=1.0, fill=False, alpha=0.10)
        ax.add_artist(circle)
        theta = r / (size[1] / 2) * 90
        if theta not in [0.0, 90.0]:
            ax.text(0, r, fr'{theta:.0f}$\degree$', color='k', fontsize=12, ha='center', va='center', alpha=0.8)
            if theta == 30.0:
                ax.text(0, r * 1.5, r'$\theta_m$', color='k', fontsize=17, ha='center', va='center', alpha=0.8)

    # Draw radial lines, the first one is the x-axis
    for phi in np.linspace(0, 2 * np.pi, num_lines, endpoint=False):
        x = [0, (size[0] / 2) * np.cos(phi)]
        y = [0, (size[1] / 2) * np.sin(phi)]
        ax.plot(x, y, color='k', linestyle='dashed', linewidth=1.0, alpha=0.08)
        ax.text(x[1] * 0.95, y[1] * 0.95, f'{round(np.degrees(phi))}°', color='k', fontsize=12, ha='center',
                va='center', alpha=0.8)
        if phi == 0.0:
            ax.text(x[1] * 0.85, y[1], r'$\phi_m$', color='k', fontsize=17, ha='center', va='center', alpha=0.8)


def tone_mapping(pixels, size, cmap='BuPu', cbar=False, coord=False, cbar_label='NDF [$sr^{-1}$]'):
    min_val = np.min(pixels)
    max_val = np.max(pixels)
    normalized = (pixels - min_val) / (max_val - min_val)
    cmap = plt.get_cmap(cmap)
    mapped = cmap(normalized)

    # Plot the image
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(mapped, extent=(-size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2), origin='lower')

    # Customize the plot for better aesthetics
    ax.set_xlim(-size[0] / 2, size[0] / 2)
    ax.set_ylim(-size[1] / 2, size[1] / 2)

    # Remove the axis
    ax.axis('off')

    # Plot the color bar
    if cbar:
        # Create a custom axes for the color bar
        cbar_ax = fig.add_axes([0.98, 0.01, 0.012, 0.28])  # [left, bottom, width, height]
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(cbar_label, rotation=270, labelpad=25, va='bottom', fontsize=14)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

    if coord:
        draw_polar_grid(ax, size)

    plt.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002, wspace=0.1, hspace=0.0)
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tone mapping measurement')
    parser.add_argument('--input', type=str, help='Input EXR file')
    parser.add_argument('--layer', type=str, help='Layer of EXR to plot', default=None)
    parser.add_argument('--channel', type=str, help='Channel to plot')
    parser.add_argument('--cmap', type=str, help='Color map to use', default='BuPu')
    parser.add_argument('--coord', action='store_true', help='Plot the coordinate system')
    parser.add_argument('--cbar', action='store_true', help='Plot the color bar')
    parser.add_argument('--save', type=str, help='Save the plot to a file')
    args = parser.parse_args()

    # Read the EXR file
    exr_file = OpenEXR.InputFile(args.input)
    print(exr_file)
    print(exr_file.header())
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    # Load values from the EXR file
    values = load_exr_channel(exr_file, args.channel)
    # Close the EXR file
    exr_file.close()

    # Plot the tone mapping
    fig, ax = tone_mapping(values, size, args.cmap, args.cbar, args.coord)

    if args.save:
        # save the plot to pdf
        fig.savefig(args.save, format='pdf', bbox_inches='tight')
    else:
        plt.show()
