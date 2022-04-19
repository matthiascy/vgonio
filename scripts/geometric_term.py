import sys
import os
import numpy as np
import seaborn as sb
import pandas as pd
from scipy.special import erf
from matplotlib import pyplot as plt


# The shadowing-masking term G depends on the distribution function D
# and the details of the micro-surface.


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


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: {} <measurement_file> <surface_name> <roughness_beckmann> <roughness_ggx>".format(os.path.basename(sys.argv[0])),
              file=sys.stderr)
        exit(-1)

    sb.set_theme(context="paper", style="whitegrid", palette="deep", font_scale=1.5)

    filename = sys.argv[1]
    surface_name = sys.argv[2]
    a_beckmann = float(sys.argv[3])
    a_ggx = float(sys.argv[4])

    df = pd.read_csv(filename, header=0, dtype=np.float32, engine='c', delim_whitespace=True)
    phis = df.phi.unique()
    thetas = df.theta.unique()

    # Plot the shadowing/masking term of different azimuthal angles
    for i in range(0, len(phis) // 36):
        phi_0 = phis[i]
        phi_1 = phis[i] + 180.0
        # Calculate the shadowing/masking term of GGX-Smith and Beckmann-Smith
        ratio_G_ggx_smith = np.zeros(len(thetas))
        ratio_G_beckmann_smith = np.zeros(len(thetas))
        for j, theta in enumerate(thetas):
            theta_rad = np.radians(theta)
            phi_rad = np.radians(phi_0)
            sin_theta = np.sin(theta_rad)
            cos_theta = np.cos(theta_rad)
            sin_phi = np.sin(phi_rad)
            cos_phi = np.cos(phi_rad)
            n = np.array([0, 0, 1])
            m = np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta])
            ratio_G_ggx_smith[j] = geometric_term_ggx_smith(n, m, m, a_ggx)
            ratio_G_beckmann_smith[j] = geometric_term_beckmann_smith(n, m, m, a_beckmann, approx=False)
        thetas_ = np.hstack((-thetas, thetas))
        df_G_ggx_smith = pd.DataFrame({
            'theta': thetas_,
            'ratio': np.tile(ratio_G_ggx_smith, 2)
        })
        df_G_beckmann_smith = pd.DataFrame({
            'theta': thetas_,
            'ratio': np.tile(ratio_G_beckmann_smith, 2)
        })
        df_G_ggx_smith.sort_values(by='theta', inplace=True)
        df_G_beckmann_smith.sort_values(by='theta', inplace=True)

        # select the data for the current phi
        df_slice = df[(df.phi == phi_0) | (df.phi == phi_1)]
        # negate the theta values for phi greater than 180
        df_slice.loc[df_slice.phi > 180.0, 'theta'] *= -1.0
        df_slice.sort_values(by=['theta'], inplace=True)
        # plot the masking function
        fig, ax = plt.subplots()
        ax.axhline(y=0.0, color='0.8')
        ax.axvline(x=0.0, color='0.8')
        ax.set_xlabel(r'$\theta$')
        ax.set_ylabel(r'$G(\mathbf{v}, \mathbf{m})$')
        ax.set_title(f"Masking-shadowing function [{surface_name}]")
        ax.set_xticks(np.arange(-90, 91, 30))
        ax.set_xticklabels([r'${}^\circ$'.format(x) for x in np.arange(-90, 91, 30)])
        ax.annotate(r'$\phi = {}^\circ$'.format(phi_0), xy=(0.0, 0.0), xycoords='axes fraction', xytext=(0.1, 0.3),
                    bbox=dict(boxstyle='round', fc='none', ec='gray'))
        ax.annotate(r'$\phi = {}^\circ$'.format(phi_1), xy=(0.0, 0.0), xycoords='axes fraction', xytext=(0.7, 0.3),
                    bbox=dict(boxstyle='round', fc='none', ec='gray'))
        ax.plot(df_G_beckmann_smith.theta, df_G_beckmann_smith.ratio, '--r',
                label=r'beckmann-smith $\alpha$ = {}'.format(a_beckmann))
        ax.plot(df_G_ggx_smith.theta, df_G_ggx_smith.ratio, '--g', label=r'ggx-smith $\alpha$ = {}'.format(a_ggx))
        ax.plot(df_slice.theta, df_slice.ratio, '-b', label='measured')
        legend = ax.legend(loc='lower center')
        fig.savefig(f'geometric_term_{surface_name}-phi_{phi_0}_{phi_1}-ab_{a_beckmann}-ag_{a_ggx}.pdf')

    # # Plot average of all values
    # averaged = df.groupby('theta').mean('ratio')
    # ax = sb.lineplot(x='theta', y='ratio', data=averaged)
    # ax.set_xlabel(r'$\theta$')
    # ax.set_ylabel(r'$G(\mathbf{v}, \mathbf{m})$')
    # ax.set_title("Measured geometric term (averaged)")
    # plt.show()
