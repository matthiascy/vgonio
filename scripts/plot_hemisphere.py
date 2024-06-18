import numpy as np
import matplotlib.pyplot as plt


def new_hemisphere_figure(elev=45, azim=-30, with_surface=False, with_axes=True, c='g', sc='b', opacity=0.3,
                          x_axis=True, y_axis=True, z_axis=True,
                          x_axis_label='X', y_axis_label='Y', z_axis_label='Z',
                          x_axis_linestyle='k--', y_axis_linestyle='k--', z_axis_linestyle='k--',
                          x_axis_length=1.0, y_axis_length=1.0, z_axis_length=1.0,
                          arrow_length=0.3,
                          arrow_length_ratio=0.3,
                          annotate=False, hide_hemisphere=False):
    """Create a new hemisphere figure.

    elev: elevation angle in the x-y plane.
    azim: azimuthal angle in the x-y plane.
    with_surface: draw a surface at the bottom centre of the hemisphere.
    with_axes: draw the x, y, and z axes.
    c: colour of the hemisphere.
    sc: colour of the surface.
    """
    # Set colours and render
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([0.0, 1.0])
    ax.set_aspect("equal")
    ax.set_proj_type('ortho')

    if not hide_hemisphere:
        r = 1
        # Create a hemisphere
        theta, phi = np.mgrid[0.0:np.pi / 2:100j, 0.0:2.0 * np.pi:100j]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color=c, alpha=opacity, linewidth=0)

    ax.view_init(elev=elev, azim=azim)

    need_to_draw_axis = [x or with_axes for x in [x_axis, y_axis, z_axis]]

    if need_to_draw_axis[0]:
        ax.plot([-x_axis_length, x_axis_length], [0, 0], [0, 0], x_axis_linestyle, alpha=0.7)  # x-axis
        ax.quiver(x_axis_length, 0, 0, arrow_length, 0, 0, color='k',
                  arrow_length_ratio=arrow_length_ratio)  # arrowhead
        ax.text(1.4, 0, 0, x_axis_label, fontsize=15, color='k')
    if need_to_draw_axis[1]:
        ax.plot([0, 0], [-y_axis_length, y_axis_length], [0, 0], y_axis_linestyle, alpha=0.7)  # y-axis
        ax.quiver(0, y_axis_length, 0, 0, arrow_length, 0, color='k',
                  arrow_length_ratio=arrow_length_ratio)  # arrowhead
        ax.text(0, 1.4, 0, y_axis_label, fontsize=15, color='k')
    if need_to_draw_axis[2]:
        ax.plot([0, 0], [0, 0], [0, z_axis_length], z_axis_linestyle, alpha=0.7)  # z-axis
        ax.quiver(0, 0, z_axis_length, 0, 0, arrow_length, color='k',
                  arrow_length_ratio=arrow_length_ratio)  # arrowhead
        ax.text(0, 0, z_axis_length * 1.4, z_axis_label, fontsize=15, color='k')

    if with_surface:
        x = np.outer(np.linspace(-0.28, 0.28, 20), np.ones(20))
        y = x.copy().T
        z = (np.sin(x ** 2) + np.cos(y ** 2)) / 4 - 0.25
        ax.plot_surface(x, y, z, color=sc, alpha=0.3, linewidth=0)

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
