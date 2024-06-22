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
    values = np.reshape(values, (size[1], size[0]))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tone mapping measurement')
    parser.add_argument('--input', type=str, help='Input EXR file')
    parser.add_argument('--channel', type=str, help='Channel to plot')
    parser.add_argument('--cmap', type=str, help='Color map to use', default='BuPu')
    parser.add_argument('--coord', action='store_true', help='Plot the coordinate system')
    parser.add_argument('--cbar', action='store_true', help='Plot the color bar')
    parser.add_argument('--save', type=str, help='Save the plot to a file')
    args = parser.parse_args()

    # Read the EXR file
    exr_file = OpenEXR.InputFile(args.input)
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Load values from the EXR file
    values = load_exr_channel(exr_file, args.channel)

    # Close the EXR file
    exr_file.close()

    min_val = np.min(values)
    max_val = np.max(values)
    normalized_values = (values - min_val) / (max_val - min_val)

    # Apply a color map
    cmap = plt.get_cmap(args.cmap)
    mapped = cmap(normalized_values)

    # Plot the image
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(mapped, extent=(-size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2), origin='lower')

    # Plot the color bar
    if args.cbar:
        # Create a custom axes for the color bar
        cbar_ax = fig.add_axes([0.98, 0.01, 0.012, 0.28])  # [left, bottom, width, height]
        norm = plt.Normalize(vmin=min_val, vmax=max_val)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(r'NDF [$sr^{-1}]$', rotation=270, labelpad=25, va='bottom', fontsize=14)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

    if args.coord:
        draw_polar_grid(ax, size)

    # Customize the plot for better aesthetics
    ax.set_xlim(-size[0] / 2, size[0] / 2)
    ax.set_ylim(-size[1] / 2, size[1] / 2)

    # Remove the axis
    ax.axis('off')

    plt.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002, wspace=0.1, hspace=0.0)

    if args.save:
        # save the plot to pdf
        plt.savefig(args.save, format='pdf', bbox_inches='tight')
    else:
        plt.show()
