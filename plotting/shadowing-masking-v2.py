"""
The shadowing-masking term G depends on the distribution function D
and the details of the micro-surface.
"""
import argparse
import re
import sys
import os
import numpy as np
import seaborn as sb
import pandas as pd
from scipy.special import erf
from matplotlib import pyplot as plt


def chi_plus(a):
    """
    Calculate the positive characteristic function which equals one if
    a > 0 and zero otherwise.

    Parameters
    ----------
    a : float
        The micro-surface roughness.

    Returns
    -------
    float
        The chi_plus function.
    """
    return 1.0 if a > 0.0 else 0.0


def geometric_term_beckmann_smith(n, m, v, a, approx=False):
    """
    Calculate the Smith shadowing-masking function with Beckmann distribution.

    The formula is given in the paper:

    "Microfacet Models for Refraction through Rough Surfaces"

    by B. Walter, S. R. Marschner, H. Li, and K. E. Torrance.

    Parameters
    ----------
    n : ndarray with dimension 3
        The macro-surface normal vector.
    m : ndarray with dimension 3
        The micro-surface normal vector of interest.
    v : ndarray with dimension 3
        The view direction vector.
    a : float
        The micro-surface roughness.
    approx : bool
        If True, calculate the approximation of the Smith shadowing-masking function.

    Returns
    -------
    float
        The Smith shadowing-masking function with Beckmann micro-facets distribution.
    """
    v_dot_n = np.dot(v, n)
    v_dot_m = np.dot(v, m)
    chi = chi_plus(v_dot_m / v_dot_n)
    tan_theta_v = np.sqrt(v[0] * v[0] + v[1] * v[1]) / v[2]  # tan(theta_v) = x^2 + y^2 / z^2
    alpha = 1.0 / (a * tan_theta_v)

    if approx:
        if alpha < 1.6:
            alpha_sqr = alpha * alpha
            return chi * (3.535 * alpha + 2.181 * alpha_sqr) / (1.0 + 2.276 * a + 2.577 * alpha_sqr)
        else:
            return chi * 1.0
    else:
        return chi * 2 / (1.0 + erf(alpha) + np.exp(-alpha * alpha) / (np.sqrt(np.pi) * alpha))


def geometric_term_ggx_smith(n, m, v, a):
    """
    Calculate the Smith shadowing-masking function with GGX distribution.

    The formula is given in the paper:

    "Microfacet Models for Refraction through Rough Surfaces"

    by B. Walter, S. R. Marschner, H. Li, and K. E. Torrance.

    Parameters
    ----------
    n : ndarray with dimension 3
        The macro-surface normal vector.
    m : ndarray with dimension 3
        The micro-surface normal vector of interest.
    v : ndarray with dimension 3
        The view direction vector.
    a : float
        The micro-surface roughness.

    Returns
    -------
    float
        The Smith shadowing-masking function with GGX micro-facets distribution.
    """
    v_dot_n = np.dot(v, n)
    v_dot_m = np.dot(v, m)
    chi = chi_plus(v_dot_m / v_dot_n)
    a_sqr = a * a
    tan_theta_v_sqr = (v[0] * v[0] + v[1] * v[1]) / (v[2] * v[2])
    return chi * 2.0 / (1.0 + np.sqrt(1.0 + a_sqr * tan_theta_v_sqr))

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        # first group is the bin size, second group is the bin count
        pattern = re.compile(r"^.*:\s([\d|.]+)°.*:\s(\d+)$")
        # todo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Microfacet shadowing/masking term plotting")
    parser.add_argument("filename", help="The file to read the data from.")
    args = parser.parse_args()
    sb.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.5)

    # todo
