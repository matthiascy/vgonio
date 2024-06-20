import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, Arc, FancyArrowPatch
import mpl_toolkits.mplot3d.art3d as art3d

from ..utils import rotate, path_patch_2d_to_3d
from ..hemisphere import hemi_coord_figure


def solid_angle_rotation_matrix(theta, phi):
    return np.matmul(rotate(phi, 'z'), rotate(theta, 'y'))


def draw_solid_angle(ax, theta, phi, c='b', r=0.12, text=r'$\omega$', annotate=False):
    # Draw the disk on the sphere
    circle = Circle((0, 0), r, color=c, alpha=0.35, linewidth=1.2)
    ax.add_patch(circle)
    m = solid_angle_rotation_matrix(theta, phi)
    path_patch_2d_to_3d(circle, m, z=1)
    circle_i_upper_pnt = (-r, 0.0, 1)
    circle_i_lower_pnt = (r, 0.0, 1)
    circle_i_upper_pnt_3d = np.matmul(m, circle_i_upper_pnt)
    circle_i_lower_pnt_3d = np.matmul(m, circle_i_lower_pnt)
    # Draw the line from the centre to the upper point
    ax.plot([0, circle_i_upper_pnt_3d[0]], [0, circle_i_upper_pnt_3d[1]], [0, circle_i_upper_pnt_3d[2]], 'k',
            alpha=0.25)
    # Draw the line from the centre to the lower point
    ax.plot([0, circle_i_lower_pnt_3d[0]], [0, circle_i_lower_pnt_3d[1]], [0, circle_i_lower_pnt_3d[2]], 'k',
            alpha=0.25)
    if annotate:
        ax.text(circle_i_upper_pnt_3d[0], circle_i_upper_pnt_3d[1], circle_i_upper_pnt_3d[2], text, fontsize=15,
                color='k')


def draw_angle_auxiliary_line(ax, dir, c='k', alpha=0.25):
    # Draw the line from the end of the direction to the xy-plane
    (x, y, z) = dir
    ax.plot([x, x], [y, y], [z, 0], c, alpha=alpha, linestyle='--')
    # Draw the line from the centre to the point on the xy-plane
    ax.plot([0, x], [0, y], [0, 0], c, alpha=alpha, linestyle='--')


def draw_direction_angle_auxiliary(ax, dir, theta, phi, theta_radius=1.0, phi_radius=1.0,
                                   azm_text=r'$\phi$',
                                   azm_text_offset=(0.2, 0.3, 0.0),
                                   zen_text=r'$\theta$',
                                   zen_text_offset=(0.0, 0.3, 0.0),
                                   annotate=True):
    x, y, z = dir
    draw_angle_auxiliary_line(ax, dir)
    # Draw the arc from the x-axis to the line on the xy-plane
    arc_phi = Arc((0, 0), width=phi_radius, height=phi_radius, angle=0, theta1=0, theta2=np.degrees(phi),
                  color='k', linestyle='dashdot', linewidth=1.5, alpha=0.8)
    ax.add_patch(arc_phi)
    art3d.pathpatch_2d_to_3d(arc_phi, z=0)
    # Draw the end of the arc
    start_angle_phi = phi - np.radians(5)
    end_angle_phi = phi
    start_x_phi = phi_radius / 2 * np.cos(start_angle_phi)
    start_y_phi = phi_radius / 2 * np.sin(start_angle_phi)
    end_x_phi = phi_radius / 2 * np.cos(end_angle_phi)
    end_y_phi = phi_radius / 2 * np.sin(end_angle_phi)
    arrow_phi = FancyArrowPatch((start_x_phi, start_y_phi), (end_x_phi, end_y_phi), mutation_scale=400,
                                color='k', arrowstyle='->', alpha=0.8, linewidth=1.5, connectionstyle="arc3,rad=0.0",
                                fill=True)
    ax.add_patch(arrow_phi)
    art3d.pathpatch_2d_to_3d(arrow_phi, z=0)
    if annotate:
        # Draw the azimuthal angle annotation
        ax.text(x + azm_text_offset[0], y + azm_text_offset[1], z + azm_text_offset[2], azm_text, fontsize=15,
                color='k', alpha=1.0)

    # Draw the arc from the z-axis to the direction
    arc_theta = Arc((0, 0), width=theta_radius, height=theta_radius, angle=0, theta1=np.degrees(np.pi / 2 - theta),
                    theta2=np.degrees(np.pi / 2), color='k', linestyle='dashdot', linewidth=1.5, alpha=0.8)
    ax.add_patch(arc_theta)
    path_patch_2d_to_3d(arc_theta, m=np.matmul(rotate(phi, 'z'), rotate(np.pi / 2, 'x')), z=0)
    # Draw the end of the arc
    start_angle_theta = (np.pi * 0.5 - theta) + np.radians(5)
    end_angle_theta = (np.pi * 0.5 - theta)
    start_x_theta = theta_radius / 2 * np.cos(start_angle_theta)
    start_y_theta = theta_radius / 2 * np.sin(start_angle_theta)
    end_x_theta = theta_radius / 2 * np.cos(end_angle_theta)
    end_y_theta = theta_radius / 2 * np.sin(end_angle_theta)
    arrow_theta = FancyArrowPatch((start_x_theta, start_y_theta), (end_x_theta, end_y_theta), mutation_scale=400,
                                  color='k', arrowstyle='->', alpha=0.8, linewidth=1.5, connectionstyle="arc3,rad=0.0",
                                  fill=True)
    ax.add_patch(arrow_theta)
    path_patch_2d_to_3d(arrow_theta, m=np.matmul(rotate(phi, 'z'), rotate(np.pi / 2, 'x')), z=0)
    if annotate:
        # Draw the zenith angle annotation
        ax.text(x + zen_text_offset[0], y + zen_text_offset[1], z + zen_text_offset[2], zen_text, fontsize=15,
                color='k', alpha=1.0)


def draw_direction(ax, theta, phi, with_solid_angle=True, annotate=False, auxiliary=False, c='g', phi_aux_radius=1.0,
                   theta_aux_radius=1.0, text=r'$\mathbf{d}$', theta_aux_text=r'$\theta$', phi_aux_text=r'$\phi',
                   phi_aux_text_offset=(0.2, 0.3, 0.0), theta_aux_text_offset=(0.0, 0.3, 0.0),
                   solid_angle_text=r'$\omega$', ):
    """
    Draw a direction and it's reflected direction on the hemisphere.

    theta: the zenith angle of the direction.
    phi: the azimuthal angle of the direction.
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    # Draw the direction
    ax.quiver(0, 0, 0, x, y, z, color=c, arrow_length_ratio=0.05, alpha=0.8, linewidth=2.0)
    if annotate:
        ax.text(x / 2, y / 2, z / 2 + 0.15, text, fontsize=15, color=c, alpha=1.0)
    # Draw the solid angle, which is the area of the patch on the sphere
    if with_solid_angle:
        draw_solid_angle(ax, theta, phi, c=c, text=solid_angle_text, annotate=annotate)

    # Draw the line annotating the angle
    if auxiliary:
        draw_direction_angle_auxiliary(ax, (x, y, z), theta, phi, theta_radius=theta_aux_radius,
                                       phi_radius=phi_aux_radius,
                                       azm_text=phi_aux_text, azm_text_offset=phi_aux_text_offset,
                                       zen_text=theta_aux_text, zen_text_offset=theta_aux_text_offset,
                                       annotate=annotate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", action="store_true", help="Generate figures")
    parser.add_argument("--annotate", action="store_true", help="Enable annotations")
    parser.add_argument("--solid", action="store_true", help="Draw the solid angle")
    parser.add_argument("--aux", action="store_true", help="Draw auxiliary lines")

    args = parser.parse_args()

    sns.set_theme(style="whitegrid", color_codes=True)

    fig, ax = hemi_coord_figure(elev=25, azim=-40, surf=False, hc='c', ha=0.15, annotate=args.annotate)

    draw_direction(ax, np.pi / 4, np.pi / 4, with_solid_angle=args.solid, annotate=args.annotate,
                   auxiliary=args.aux, phi_aux_radius=1.0, theta_aux_radius=0.8, text=r'$\mathbf{i}$',
                   theta_aux_text=r'$\theta_i$', phi_aux_text=r'$\phi_i$',
                   theta_aux_text_offset=(-0.4, -0.45, -np.cos(np.pi / 4) * 0.35),
                   phi_aux_text_offset=(0.1, -0.4, -np.cos(np.pi / 4)),
                   solid_angle_text=r'$\omega_i$', c='r')
    draw_direction(ax, np.pi / 5, np.pi / 3 + np.pi, with_solid_angle=args.solid, annotate=args.annotate,
                   auxiliary=args.aux, phi_aux_radius=0.7, theta_aux_radius=0.7, text=r'$\mathbf{o}$',
                   theta_aux_text=r'$\theta_o$', phi_aux_text=r'$\phi_o$',
                   theta_aux_text_offset=(0.2, 0.35, -np.cos(np.pi / 5) * 0.45),
                   phi_aux_text_offset=(0.45, 0.85, -np.cos(np.pi / 5)),
                   solid_angle_text=r'$\omega_o$', c='m')

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
    if args.gen:
        plt.savefig("spherical-coord.pdf")
    else:
        plt.show()
