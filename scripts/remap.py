# Read two PNG files and remap them to the same scale and color map
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image


def read_image(file_path, is_depth=False):
    # Open the image file
    img = Image.open(file_path)
    # Convert image to a numpy array and get only the first channel
    if is_depth:
        img_array = np.array(img)[:, :]
    else:
        img_array = np.array(img)[:, :, 0]
    return img_array


def normalize_image(img, global_min, global_max):
    # Normalize the image to the range [0, 1]
    norm_img = (img - global_min) / (global_max - global_min)
    return norm_img


def plot_map_comparison(dmap, tmap, vmap, colormap='viridis', save_sep=False):
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    fig_sep, ax_sep = plt.subplots(figsize=(8, 8))
    ax_sep.axis('off')

    # Plot the depth map
    im1 = axs[0].imshow(dmap, cmap='gray')
    # , cmap=colormap, norm=mcolors.Normalize(vmin=dmap.min(), vmax=dmap.max())
    axs[0].set_title('Depth Map', fontsize=24)
    axs[0].axis('off')

    # Find global min and max values of tmap and vmap
    global_min = min(tmap.min(), vmap.min())
    global_max = max(tmap.max(), vmap.max())

    print("Global min: ", global_min, "Global max: ", global_max)

    # Plot the total area map
    im2 = axs[1].imshow(tmap, cmap=colormap, norm=mcolors.Normalize(vmin=global_min, vmax=global_max))
    if save_sep:
        ax_sep.imshow(tmap, cmap=colormap, norm=mcolors.Normalize(vmin=global_min, vmax=global_max))
        # plt.imsave('gaf-total-area.pdf', tmap, cmap=colormap)
        fig_sep.savefig('gaf-total-area.pdf', bbox_inches='tight', pad_inches=0.1)
    axs[1].set_title('Total Area', fontsize=24)
    axs[1].axis('off')

    # Plot the visible area map
    im3 = axs[2].imshow(vmap, cmap=colormap, norm=mcolors.Normalize(vmin=global_min, vmax=global_max))
    if save_sep:
        # plt.imsave('gaf-visible-area.pdf', vmap, cmap=colormap)
        ax_sep.imshow(vmap, cmap=colormap, norm=mcolors.Normalize(vmin=global_min, vmax=global_max))
        fig_sep.savefig('gaf-visible-area.pdf', bbox_inches='tight', pad_inches=0.1)
    axs[2].set_title('Visible Area', fontsize=24)
    axs[2].axis('off')

    # Plot the difference between the two images
    diff = np.abs(vmap - tmap)
    im4 = axs[3].imshow(diff, cmap=colormap, norm=mcolors.Normalize(vmin=global_min, vmax=global_max), resample=True)
    if save_sep:
        # update the values to max if the difference is not zero
        # diff[diff > 0] = global_max
        # plt.imsave('gaf-difference.pdf', diff, cmap=colormap)
        ax_sep.imshow(diff, cmap=colormap, norm=mcolors.Normalize(vmin=global_min, vmax=global_max))
        fig_sep.savefig('gaf-difference.pdf', bbox_inches='tight', pad_inches=0.1)
    axs[3].set_title('Difference', fontsize=24)
    axs[3].axis('off')

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.1, hspace=0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remap two images to the same scale and colormap')
    parser.add_argument('--id', type=str, help='Image stores the depth')
    parser.add_argument('--it', type=str, help='Image stores the total area')
    parser.add_argument('--iv', type=str, help='Image stores the visible area')
    parser.add_argument('--cmap', type=str, help='Colormap to use', default='viridis')
    parser.add_argument('--save', type=str, help='Save the plot to a file')
    parser.add_argument('--save-sep', action='store_true', help='Save the images separately')
    args = parser.parse_args()

    dmap = read_image(args.id, is_depth=True)
    tmap = read_image(args.it)
    vmap = read_image(args.iv)

    # Plot images with the same colormap
    plot_map_comparison(dmap, tmap, vmap, colormap=args.cmap, save_sep=args.save_sep)

    if args.save:
        plt.savefig(args.save, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
