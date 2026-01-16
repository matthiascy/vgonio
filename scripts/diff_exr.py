import argparse

import Imath
import OpenEXR
import cv2
import numpy as np


def load_exr_image(filename):
    """ Load an EXR image and return it as a NumPy array (H, W, C). """
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = ['R', 'G', 'B']
    pixel_type = Imath.PixelType(Imath.PixelType.HALF)

    img = np.zeros((height, width, len(channels)), dtype=np.float32)

    for i, channel in enumerate(channels):
        raw_data = exr_file.channel(channel, pixel_type)
        img[:, :, i] = np.frombuffer(raw_data, dtype=np.float16).astype(np.float32).reshape(height, width)

    return img


def compute_difference(img1, img2, discard_negative=True):
    """ Compute the difference between two EXR images. """
    diff = img1 - img2
    if discard_negative:
        diff[diff < 0] = 0
    else:
        diff = np.abs(diff)
    return diff


def save_difference_as_image(diff, output_file):
    """ Save the difference as an image (PNG for easy viewing). """
    diff_normalized = (diff / np.max(diff) * 255).astype(np.uint8)  # Normalize
    cv2.imwrite(output_file, cv2.cvtColor(diff_normalized, cv2.COLOR_RGB2BGR))  # Save


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Difference images')
    parser.add_argument('-a', help='First image', required=True)
    parser.add_argument('-b', help='Second image', required=True)
    parser.add_argument('-o', help='Output file', required=True)
    args = parser.parse_args()

    # Load two EXR images
    img1 = load_exr_image(args.a)
    img2 = load_exr_image(args.b)

    # Compute absolute difference
    diff_abs = compute_difference(img1, img2, False)
    # diff_a_minus_b = compute_difference(img1, img2, True)
    # diff_b_minus_a = compute_difference(img1, img2, True)

    # Save as a PNG for easy viewing
    save_difference_as_image(diff_abs, f"{args.o}-abs.png")
    # save_difference_as_image(diff_a_minus_b, f"{args.o}-a_minus_b.png")
    # save_difference_as_image(diff_b_minus_a, f"{args.o}-b_minus_a.png")
