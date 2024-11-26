import argparse

import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_exr_channel(exr_file, channel, size):
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


def draw_polar_grid(ax, size, pstep=45.0, tstep=30.0, color='k', ac='m'):
    num_lines = int(360 / pstep)
    num_circles = int(90 / tstep) + 1
    print(f"Drawing polar grid with {num_lines} radial lines and {num_circles} circles")
    # Draw circles, each representing a certain angle
    # Do not draw the text of the angle if it is 0 or 90 degrees
    for theta in np.linspace(0, 90, num_circles):
        r = size[1] / 2 * 2 * np.sin(np.radians(theta) / 2) / np.sqrt(2)
        # for r in np.linspace(0, size[1] / 2, num_circles):
        circle = plt.Circle((0, 0), r, color=color, linestyle='-', linewidth=1.0, fill=False, alpha=0.10)
        ax.add_artist(circle)
        if theta not in [0.0, 90.0]:
            ax.text(0, r, fr'{theta:.0f}$\degree$', color=color, fontsize=12, ha='center', va='center', alpha=0.8)
            if theta == 30.0:
                if tstep < 30.0:
                    ax.text(-20.0, r * 1.5, fr'$\theta_{ac}$', color=color, fontsize=17, ha='center', va='center',
                            alpha=0.8)
                else:
                    ax.text(0, r * 1.5, fr'$\theta_{ac}$', color=color, fontsize=17, ha='center', va='center',
                            alpha=0.8)

    # Draw radial lines, the first one is the x-axis
    for phi in np.linspace(0, 2 * np.pi, num_lines, endpoint=False):
        x = [0, (size[0] / 2) * np.cos(phi)]
        y = [0, (size[1] / 2) * np.sin(phi)]
        ax.plot(x, y, color=color, linestyle='dashed', linewidth=1.0, alpha=0.08)
        ax.text(x[1] * 0.95, y[1] * 0.95, f'{round(np.degrees(phi))}°', color=color, fontsize=12, ha='center',
                va='center', alpha=0.8)
        if phi == 0.0:
            ax.text(x[1] * 0.85, y[1], fr'$\phi_{ac}$', color=color, fontsize=17, ha='center', va='center', alpha=0.8)


def tone_mapping(pixels, size, cmap='BuPu', cbar=False, coord=False, cbar_label='NDF [$sr^{-1}$]', color='k',
                 pstep=45.0, tstep=30.0, ac='m'):
    min_val = np.min(pixels)
    max_val = np.max(pixels)
    normalized = (pixels - min_val) / (max_val - min_val)
    cmap = plt.get_cmap(cmap)
    mapped = cmap(normalized)

    # Plot the image
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(mapped, extent=(-size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2), origin='upper')

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
        cbar.set_label(cbar_label, rotation=270, labelpad=25, va='bottom', fontsize=14, color=color)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.tick_params(colors=color)

    if coord:
        draw_polar_grid(ax, size, color=color, pstep=pstep, tstep=tstep, ac=ac)

    plt.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002, wspace=0.1, hspace=0.0)
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tone mapping measurement')
    parser.add_argument('--input', type=str, nargs='+', help='Input EXR file')
    parser.add_argument('--layer', type=str, help='Layer of EXR to plot', default=None)
    parser.add_argument('--channel', type=str, help='Channel to plot')
    parser.add_argument('--cmap', type=str, help='Color map to use', default='BuPu')
    parser.add_argument('--coord', action='store_true', help='Plot the coordinate system')
    parser.add_argument('--cbar', action='store_true', help='Plot the color bar')
    parser.add_argument('--save', type=str, help='Save the plot to a file')
    parser.add_argument('--diff', action='store_true', help='Plot the difference between two images')
    parser.add_argument('--fc', type=str, help='Font color', default='k')
    args = parser.parse_args()

    if args.diff:
        # Read the two EXR files each time
        exr_file1 = OpenEXR.InputFile(args.input[0])
        dw = exr_file1.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        vals1 = load_exr_channel(exr_file1, args.channel, size)
        exr_file1.close()

        exr_file2 = OpenEXR.InputFile(args.input[1])
        vals2 = load_exr_channel(exr_file2, args.channel, size)
        exr_file2.close()

        # Calculate the difference
        diff = np.abs(vals2 - vals1)
        # Calculate the mean squared error
        mse = np.mean(diff ** 2)

        fig, ax = tone_mapping(diff, size, args.cmap, args.cbar, args.coord, cbar_label='Difference', color=args.fc)
        ax.text(-size[0] / 2 + 50, size[1] / 2 - 10, f'MSE: {mse:.4f}', color=args.fc, fontsize=14, ha='center',
                va='center', alpha=0.8)

        if args.save:
            fig.savefig(args.save, format='pdf', bbox_inches='tight')
        else:
            plt.show()

    else:
        for input in args.input:
            # Read the EXR file
            exr_file = OpenEXR.InputFile(input)
            # print(exr_file)
            # print(exr_file.header())
            dw = exr_file.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            # Load values from the EXR file
            values = load_exr_channel(exr_file, args.channel, size)
            # Close the EXR file
            exr_file.close()

            # Plot the tone mapping
            fig, ax = tone_mapping(values, size, args.cmap, args.cbar, args.coord, color=args.fc)

            if args.save:
                if args.save.endswith('.pdf'):
                    fig.savefig(args.save, format='pdf', bbox_inches='tight')
                else:
                    fig.savefig(args.save, format='png', bbox_inches='tight', dpi=100)
            else:
                plt.show()
