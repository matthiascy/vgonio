import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from matplotlib.patches import Circle, Arc, FancyArrowPatch
import mpl_toolkits.mplot3d.art3d as art3d

from vgplt.hemisphere import hemi_coord_figure
from vgplt.utils import rotate, path_patch_2d_to_3d


def plot_sphere(ax):
    r = 0.1
    # Create a hemisphere
    theta, phi = np.mgrid[0.0:np.pi:100j, 0.0:2.0 * np.pi:100j]
    x = r * np.sin(theta) * np.cos(phi) - 0.5
    y = r * np.sin(theta) * np.sin(phi) - 0.5
    z = r * np.cos(theta) + 0.8
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='orange', alpha=0.8, linewidth=0, shade=False)


def plot_cylinder(ax, mtheta, mphi, radius=0.05, height=0.2, center=(0, 0, 1), c='orange'):
    m = np.matmul(rotate(mphi, 'z'), rotate(mtheta, 'y'))

    (x_center, y_center, z_center) = center

    # Define the cylinder
    z = np.linspace(z_center - height / 2, z_center + height / 2, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta) + x_center
    y = radius * np.sin(theta) + y_center

    # Flatten the coordinates
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    # Stack the coordinates and apply the rotation matrix
    xyz = np.vstack((x_flat, y_flat, z_flat))
    rotated = np.matmul(m, xyz)

    # Reshape the rotated coordinates back to the original shape
    x_rot = rotated[0].reshape(x.shape)
    y_rot = rotated[1].reshape(y.shape)
    z_rot = rotated[2].reshape(z.shape)

    # Create the top and bottom caps of the cylinder
    theta_cap = np.linspace(0, 2 * np.pi, 50)
    x_cap = radius * np.cos(theta_cap)
    y_cap = radius * np.sin(theta_cap)
    z_cap_top = np.full_like(theta_cap, z_center + height / 2)
    z_cap_bot = np.full_like(theta_cap, z_center - height / 2)

    # Apply the rotation matrix to the top and bottom caps
    xyz_cap_top = np.vstack((x_cap, y_cap, z_cap_top))
    rotated_cap_top = np.matmul(m, xyz_cap_top)
    xyz_cap_bot = np.vstack((x_cap, y_cap, z_cap_bot))
    rotated_cap_bot = np.matmul(m, xyz_cap_bot)

    # Plot the cylinder
    ax.plot_surface(x_rot, y_rot, z_rot, rstride=1, cstride=1, color=c, alpha=0.5, linewidth=0, shade=False)
    ax.plot_trisurf(rotated_cap_top[0, :], rotated_cap_top[1, :], rotated_cap_top[2, :], color=c, alpha=0.8,
                    shade=True, linewidth=0)
    ax.plot_trisurf(rotated_cap_bot[0, :], rotated_cap_bot[1, :], rotated_cap_bot[2, :], color=c, alpha=0.8,
                    shade=True, linewidth=0)

    center_cylinder_bot = np.matmul(m, np.array([[0], [0], [z_center - height / 2]]))
    x_center_bot, y_center_bot, z_center_bot = center_cylinder_bot.flatten()

    # Outline the bottom of the cylinder
    ax.plot(rotated_cap_bot[0, :], rotated_cap_bot[1, :], rotated_cap_bot[2, :], 'g--')

    # Plot the movement of the cylinder
    arc_phi = Arc((0, 0), width=2.0, height=2.0, angle=0, theta1=-20,
                  theta2=20, color='k', linewidth=1, alpha=0.8, linestyle='dashdot')
    ax.add_patch(arc_phi)
    path_patch_2d_to_3d(arc_phi, m=np.matmul(rotate(mphi, 'z'), rotate(-(np.pi / 2 - mtheta), 'y')), z=0)

    start_angle_phi = -15
    end_angle_phi = -20
    start_x_phi = 1.0 * np.cos(np.radians(start_angle_phi))
    start_y_phi = 1.0 * np.sin(np.radians(start_angle_phi))
    end_x_phi = 1.0 * np.cos(np.radians(end_angle_phi))
    end_y_phi = 1.0 * np.sin(np.radians(end_angle_phi))
    arrow_phi = FancyArrowPatch((start_x_phi, start_y_phi), (end_x_phi, end_y_phi), mutation_scale=400,
                                color='k', arrowstyle='->', alpha=0.8, linewidth=1.2, connectionstyle="arc3,rad=0.0",
                                fill=False)
    ax.add_patch(arrow_phi)
    path_patch_2d_to_3d(arrow_phi, m=np.matmul(rotate(mphi, 'z'), rotate(-(np.pi / 2 - mtheta), 'y')), z=0)

    start_angle_phi = 15
    end_angle_phi = 20
    start_x_phi = 1.0 * np.cos(np.radians(start_angle_phi))
    start_y_phi = 1.0 * np.sin(np.radians(start_angle_phi))
    end_x_phi = 1.0 * np.cos(np.radians(end_angle_phi))
    end_y_phi = 1.0 * np.sin(np.radians(end_angle_phi))
    arrow_phi = FancyArrowPatch((start_x_phi, start_y_phi), (end_x_phi, end_y_phi), mutation_scale=400,
                                color='k', arrowstyle='->', alpha=0.8, linewidth=1.2, connectionstyle="arc3,rad=0.0",
                                fill=False)
    ax.add_patch(arrow_phi)
    path_patch_2d_to_3d(arrow_phi, m=np.matmul(rotate(mphi, 'z'), rotate(-(np.pi / 2 - mtheta), 'y')), z=0)

    arc_theta = Arc((0, 0), width=2.0, height=2.0, angle=0, theta1=np.degrees((np.pi / 2 - mtheta)) - 20,
                    theta2=np.degrees((np.pi / 2 - mtheta)) + 20, color='k', linewidth=1, alpha=0.8,
                    linestyle='dashdot')
    ax.add_patch(arc_theta)
    path_patch_2d_to_3d(arc_theta, m=np.matmul(rotate(mphi, 'z'), rotate(np.pi / 2, 'x')), z=0)

    start_angle_theta = np.degrees((np.pi / 2 - mtheta)) - 15
    end_angle_theta = np.degrees((np.pi / 2 - mtheta)) - 20
    start_x_theta = 1.0 * np.cos(np.radians(start_angle_theta))
    start_y_theta = 1.0 * np.sin(np.radians(start_angle_theta))
    end_x_theta = 1.0 * np.cos(np.radians(end_angle_theta))
    end_y_theta = 1.0 * np.sin(np.radians(end_angle_theta))
    arrow_theta = FancyArrowPatch((start_x_theta, start_y_theta), (end_x_theta, end_y_theta), mutation_scale=400,
                                  color='k', arrowstyle='->', alpha=0.8, linewidth=1.2, connectionstyle="arc3,rad=0.0",
                                  fill=False)
    ax.add_patch(arrow_theta)
    path_patch_2d_to_3d(arrow_theta, m=np.matmul(rotate(mphi, 'z'), rotate(np.pi / 2, 'x')), z=0)

    start_angle_theta = np.degrees((np.pi / 2 - mtheta)) + 15
    end_angle_theta = np.degrees((np.pi / 2 - mtheta)) + 20
    start_x_theta = 1.0 * np.cos(np.radians(start_angle_theta))
    start_y_theta = 1.0 * np.sin(np.radians(start_angle_theta))
    end_x_theta = 1.0 * np.cos(np.radians(end_angle_theta))
    end_y_theta = 1.0 * np.sin(np.radians(end_angle_theta))
    arrow_theta = FancyArrowPatch((start_x_theta, start_y_theta), (end_x_theta, end_y_theta), mutation_scale=400,
                                  color='k', arrowstyle='->', alpha=0.8, linewidth=1.2, connectionstyle="arc3,rad=0.0",
                                  fill=False)
    ax.add_patch(arrow_theta)
    path_patch_2d_to_3d(arrow_theta, m=np.matmul(rotate(mphi, 'z'), rotate(np.pi / 2, 'x')), z=0)

    return x_center_bot, y_center_bot, z_center_bot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gen", action="store_true", help="Generate figures")

    args = parser.parse_args()

    sns.set_theme(style="whitegrid", color_codes=True)

    fig, ax = hemi_coord_figure(elev=38, azim=-50, surf=True, c='c', sc='g', ha=0.05, arrow_length=0.1,
                                arrow_length_ratio=0.6, axes_labels=('', '', r'$\hat{n}$'), zlen=0.5)
    ax.text(0.4, -0.4, 0.0, r'specimen', fontsize=15, color='k')

    # Plot the light source
    theta = np.pi / 4
    phi = np.pi / 6 + np.pi
    (l_bot_x, l_bot_y, l_bot_z) = plot_cylinder(ax, theta, phi)
    ax.text(-0.7, -0.8, 1.2, r'light source', fontsize=15, color='k')
    ax.quiver(l_bot_x, l_bot_y, l_bot_z, -l_bot_x, -l_bot_y, -l_bot_z, color='orangered', arrow_length_ratio=0.1,
              alpha=0.8, linewidth=2.0)

    # Plot the sensor
    theta_o = np.pi / 3
    phi_o = np.pi / 4
    (l_bot_x, l_bot_y, l_bot_z) = plot_cylinder(ax, theta_o, phi_o, c='darkorchid', height=0.1, radius=0.08)
    ax.text(0.65, 0.55, 0.8, r'sensor', fontsize=15, color='k')
    ax.quiver(0, 0, 0, l_bot_x, l_bot_y, l_bot_z, color='orangered', arrow_length_ratio=0.1,
              alpha=0.8, linewidth=2.0)
    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)

    if args.gen:
        plt.savefig("gonio-concept.pdf")
    else:
        plt.show()
