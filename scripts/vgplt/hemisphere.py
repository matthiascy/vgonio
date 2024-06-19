import numpy as np
import matplotlib.pyplot as plt


def hemi_coord_figure(elev=45, azim=-30, axes='xyz', hc='g', sc='b', ha=0.3, surf=False, annotate=False,
                      hide=False, hshade=True, **kwargs):
    """
    Create a new hemisphere figure.
    :param elev: elevation angle of the view.
    :param azim: azimuthal angle of the view.
    :param surf: draw a surface at the bottom centre of the hemisphere.
    :param axes: axes to draw, 'xyz' for all.
    :param hc: colour of the hemisphere.
    :param sc: colour of the surface.
    :param ha: alpha (opacity) of the hemisphere.
    :param annotate: if True, annotate the hemisphere domain.
    :param hide: if True, hide the hemisphere.
    :param hshade: if True, shade the hemisphere.
    :param kwargs:
        * 'figsize': size of the figure.
        * 'axes_labels': labels of the axes in the order of x, y, z.
        * 'axes_alpha': alpha of the axes.
        * 'xlen': length of the x-axis.
        * 'ylen': length of the y-axis.
        * 'zlen': length of the z-axis.
        * 'xlinestyle': line style of the x-axis.
        * 'ylinestyle': line style of the y-axis.
        * 'zlinestyle': line style of the z-axis.
        * 'arrow_length': length of the arrow, default 0.3.
        * 'arrow_length_ratio': length of the arrowhead relative to the arrow, default 0.3.
    :return: fig, ax
    """
    # Set colours and render
    figsize = kwargs.get('figsize', (8, 8))
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([0.0, 1.0])
    ax.set_aspect("equal")
    ax.set_proj_type('ortho')

    if not hide:
        r = 1
        # Create a hemisphere
        theta, phi = np.mgrid[0.0:np.pi / 2:100j, 0.0:2.0 * np.pi:100j]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        ax.plot_surface(x, y, z, rstride=1, cstride=1, color=hc, alpha=ha, linewidth=0, shade=hshade)

    ax.view_init(elev=elev, azim=azim)

    axes_labels = kwargs.get('axes_labels', ['X', 'Y', 'Z'])
    xlen = kwargs.get('xlen', 1.0)
    ylen = kwargs.get('ylen', 1.0)
    zlen = kwargs.get('zlen', 1.0)
    xlinestyle = kwargs.get('xlinestyle', 'k--')
    ylinestyle = kwargs.get('ylinestyle', 'k--')
    zlinestyle = kwargs.get('zlinestyle', 'k--')
    arrow_length = kwargs.get('arrow_length', 0.3)
    arrow_length_ratio = kwargs.get('arrow_length_ratio', 0.3)
    axes_alpha = kwargs.get('axes_alpha', (0.7, 0.7, 0.7))

    if 'x' in axes:
        ax.plot([-xlen, xlen], [0, 0], [0, 0], xlinestyle, alpha=axes_alpha[0])  # x-axis
        ax.quiver(xlen, 0, 0, arrow_length, 0, 0, color='k',
                  arrow_length_ratio=arrow_length_ratio)  # arrowhead
        ax.text(1.4, 0, 0, axes_labels[0], fontsize=15, color='k')
    if 'y' in axes:
        ax.plot([0, 0], [-ylen, ylen], [0, 0], ylinestyle, alpha=axes_alpha[1])  # y-axis
        ax.quiver(0, ylen, 0, 0, arrow_length, 0, color='k',
                  arrow_length_ratio=arrow_length_ratio)  # arrowhead
        ax.text(0, 1.4, 0, axes_labels[1], fontsize=15, color='k')
    if 'z' in axes:
        ax.plot([0, 0], [0, 0], [0, zlen], zlinestyle, alpha=axes_alpha[2])  # z-axis
        ax.quiver(0, 0, zlen, 0, 0, arrow_length, color='k',
                  arrow_length_ratio=arrow_length_ratio)  # arrowhead
        ax.text(0, 0, zlen * 1.4, axes_labels[2], fontsize=15, color='k')

    if surf:
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
