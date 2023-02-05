"""
The shadowing-masking term G depends on the distribution function D
and the details of the micro-surface.
"""
import argparse
import re
import os
from io import StringIO
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        # first group is the bin size, second group is the bin count
        pattern = re.compile(r"^.*:\s([\d|.]+)°.*:\s(\d+)$")
        if lines[0] == "microfacet shadowing and masking function\n":
            azimuth_res = re.match(pattern, lines[1]).groups()
            zenith_res = re.match(pattern, lines[2]).groups()
            return np.genfromtxt(StringIO(lines[3]), dtype=np.float32, delimiter=' ') \
                .reshape((int(azimuth_res[1]), int(zenith_res[1]), int(azimuth_res[1]), int(zenith_res[1]))), \
                float(azimuth_res[0]), float(zenith_res[0])
        else:
            raise ValueError("The file is not a valid shadowing-masking data file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microfacet shadowing/masking term plotting")
    parser.add_argument("filename", help="The file to read the data from.")
    parser.add_argument("-mt", "--m-theta", type=float, help="The zenith angle (theta) of the micro-facet normal (m).")
    parser.add_argument("-mp", "--m-phi", type=float, help="The azimuthal angle (phi) of the micro-facet normal (m).")
    parser.add_argument("-p", "--phi", nargs="*", type=float, help="The azimuthal angle to plot.")
    args = parser.parse_args()
    sb.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.5)

    data, azimuth_bin_size, zenith_bin_size = read_data(args.filename)
    azimuth_bins = np.arange(0, 360, azimuth_bin_size)
    zenith_bins = np.arange(0, 90 + zenith_bin_size, zenith_bin_size)

    basename = os.path.basename(args.filename)
    output_dir = f"{basename.split('.')[0]}_plots"

    figures = []
