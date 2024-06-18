import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def new_hemisphere_figure(elev=45, azim=-30, with_surface=True, with_axes=True, color='g', opacity=0.3, annotate=False):
    """Create a new hemisphere figure.

    elev: elevation angle in the x-y plane.
    azim: azimuthal angle in the x-y plane.
    with_surface: draw a surface at the bottom centre of the hemisphere.
    with_axes: draw the x, y, and z axes.
    color: colour of the hemisphere.
    """
    # Set colours and render
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([0.0, 1.0])
    ax.set_aspect("equal")
    ax.set_proj_type('ortho')

    r = 1
    # Create a hemisphere
    theta, phi = np.mgrid[0.0:np.pi / 2:100j, 0.0:2.0 * np.pi:100j]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color=color, alpha=opacity, linewidth=0)
    ax.view_init(elev=elev, azim=azim)

    if with_axes:
        # Draw the dashed axes
        ax.plot([-1.0, 1.0], [0, 0], [0, 0], 'k--', alpha=0.7)  # x-axis
        ax.plot([0, 0], [-1.0, 1.0], [0, 0], 'k--', alpha=0.7)  # y-axis
        ax.plot([0, 0], [0, 0], [0, 1.0], 'k--', alpha=0.7)  # z-axis

        # Draw arrowheads at the end of the axes
        ax.quiver(1.0, 0, 0, 0.3, 0, 0, color='k', arrow_length_ratio=0.3)
        ax.quiver(0, 1.0, 0, 0, 0.3, 0, color='k', arrow_length_ratio=0.3)
        ax.quiver(0, 0, 1.0, 0, 0, 0.3, color='k', arrow_length_ratio=0.3)

        # Label the axes
        ax.text(1.4, 0, 0, '$X$', fontsize=15, color='k')
        ax.text(0, 1.4, 0, '$Y$', fontsize=15, color='k')
        ax.text(0, 0, 1.4, '$Z$', fontsize=15, color='k')

    if with_surface:
        x = np.outer(np.linspace(-0.28, 0.28, 20), np.ones(20))
        y = x.copy().T
        z = (np.sin(x ** 2) + np.cos(y ** 2)) / 4 - 0.25
        ax.plot_surface(x, y, z, color='b', alpha=0.3, linewidth=0)

    if annotate:
        # annotate the hemisphere domain
        ax.text(-0.8, -0.8, 0.0, r'$\Omega$', fontsize=15, color='k')

    # hide gridlines
    ax.grid(False)
    # hide y and z plane
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # hide x and z plane
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # hide axis line
    ax.xaxis.line.set_color("white")
    ax.yaxis.line.set_color("white")
    ax.zaxis.line.set_color("white")
    # hide ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    return fig, ax
