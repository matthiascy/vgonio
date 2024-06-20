import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay, ConvexHull

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d

from ..hemisphere import hemi_coord_figure
from ..utils import rotate, path_patch_2d_to_3d


# Generate grid surface with alternating peaks and valleys
def generate_valley_surface(grid_size, scale=1.0):
    x = np.linspace(-1, 1, grid_size) * scale
    y = np.linspace(-1, 1, grid_size) * scale
    x, y = np.meshgrid(x, y)
    z = (np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) / 4 + np.random.uniform(-1, 1) * 0.2) * scale
    return x, y, z


# Plot the surface and vectors
def plot_valley_surface_with_vectors():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Masking Function')

    grid_size = 20
    x, y, z = generate_valley_surface(grid_size)

    # Plot the surface
    ax.plot_surface(x, y, z, rstride=1, cstride=1, color='b', alpha=0.6, edgecolor='none')

    # Plot the vectors
    vectors = [
        {"start": [2, 2, 0.5], "dir": [1, 1, -1], "color": 'g', "label": r'$\omega_o$'},
        {"start": [2, 2, 0.5], "dir": [1, -1, -1], "color": 'y', "label": r'$\omega_i$'}
    ]
    for vec in vectors:
        ax.quiver(*vec["start"], *vec["dir"], length=1, color=vec["color"], label=vec["label"])

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()


# Calculate the normal vector from spherical coordinates
def spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


# Project points onto a plane defined by a point and a normal vector
def project_points_onto_plane(points, plane_point, plane_normal):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    points_to_plane_point = points - plane_point
    projection = points - np.outer(np.dot(points_to_plane_point, plane_normal), plane_normal)
    return projection


# Calculate normals for each triangle in the surface
def calculate_normals(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    return normals


# Find the edges of the triangulation that are on the boundary
def find_boundary_edges(triangles):
    edges = {}
    for tri in triangles:
        for i in range(3):
            edge = tuple(sorted([tri[i], tri[(i + 1) % 3]]))
            if edge in edges:
                edges[edge] += 1
            else:
                edges[edge] = 1
    boundary_edges = [edge for edge, count in edges.items() if count == 1]
    return boundary_edges


# Highlight the masked areas based on the visibility criteria
def highlight_masked_areas(ax, x, y, z, close_triangles):
    for tri in close_triangles:
        pts = np.array([x[tri], y[tri], z[tri]]).T
        poly = plt.Polygon(pts[:, :2], facecolor='green', edgecolor='black', alpha=0.3)
        ax.add_patch(poly)
        art3d.pathpatch_2d_to_3d(poly, z=pts[:, 2].min(), zdir="z")


# Plot the surface, shadow map plane, and projection shape
def plot_shadow_map_generation(ax, theta=45, phi=45, filter=True, scale=0.2, tolerance=30):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    grid_size = 10
    x, y, z = generate_valley_surface(grid_size, scale=scale)

    # Flatten the 2D arrays into 1D arrays
    x_flat = x.ravel()
    y_flat = y.ravel()
    z_flat = z.ravel()

    # Create vertices and faces for the surface
    points = np.c_[x_flat, y_flat, z_flat]
    tri = Delaunay(points[:, :2])
    faces = tri.simplices

    # Calculate normals for each triangle
    normals = calculate_normals(points, faces)

    # Define the directional vector and the tolerance for the angle
    dir_vector = spherical_to_cartesian(theta, phi)
    angle_tolerance = np.deg2rad(tolerance)  # Tolerance angle in radians

    # Filter triangles whose normals are close to the directional vector
    dot_product = np.dot(normals, dir_vector)
    angle_diff = np.arccos(np.clip(dot_product, -1.0, 1.0))
    close_triangles = faces[angle_diff < angle_tolerance]

    # Map original indices to filtered indices
    original_to_filtered_map = {original_idx: i for i, original_idx in enumerate(np.unique(close_triangles))}
    filtered_triangles = np.vectorize(original_to_filtered_map.get)(close_triangles)

    # Draw normals of filtered triangles
    if filter:
        for tri in close_triangles:
            tri_pts = points[tri]
            tri_centre = np.mean(tri_pts, axis=0)
            tri_normal = calculate_normals(tri_pts, np.array([[0, 1, 2]]))[0]
            ax.quiver(tri_centre[0], tri_centre[1], tri_centre[2],
                      tri_normal[0], tri_normal[1], tri_normal[2],
                      color='r', alpha=0.6, length=0.2)

    # Project filtered triangles onto the plane
    plane_normal = dir_vector
    plane_point = plane_normal * 1

    projected_points = None
    if filter:
        filtered_points = points[np.unique(close_triangles)]
        projected_points = project_points_onto_plane(filtered_points, plane_point, plane_normal)
    else:
        projected_points = project_points_onto_plane(points, plane_point, plane_normal)

    # Find boundary edges of the filtered triangles
    boundary_edges = find_boundary_edges(filtered_triangles) if filter else find_boundary_edges(faces)
    boundary_indices = np.unique(np.array(boundary_edges).flatten())

    # Plot a rectangle to represent the shadow map plane
    rect = plt.Rectangle((-0.5, -0.5), 1, 1, color='darkolivegreen', alpha=0.2, linewidth=0)
    ax.add_patch(rect)
    m = np.matmul(rotate(phi, 'z'), rotate(theta, 'y'))
    path_patch_2d_to_3d(rect, m=m, z=1)

    # Plot the dotted line from the centre to the plane centre
    ax.plot([0, plane_point[0]], [0, plane_point[1]], [0, plane_point[2]], 'k--', alpha=0.6)

    # Plot the surface
    ax.plot_trisurf(x_flat, y_flat, z_flat, triangles=faces, color='b', alpha=0.6, edgecolor='none')

    # Plot the exact shape formed by the outermost points of the projected triangles
    for edge in boundary_edges:
        pts = projected_points[np.array(edge)]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'k-', alpha=0.6)

    # Plot the plane normal
    ax.quiver(plane_point[0], plane_point[1], plane_point[2],
              plane_normal[0], plane_normal[1], plane_normal[2],
              linewidth=2, color='k', alpha=0.8, length=0.4)
    ax.text(plane_point[0], plane_point[1], plane_point[2] + 0.4, r'$\hat{m}$', fontsize=20, color='k')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", action="store_true", help="Generate figures")
    parser.add_argument("--theta", type=int, default=30, help="Zenith angle in degrees")
    parser.add_argument("--phi", type=int, default=45, help="Azimuthal angle in degrees")
    parser.add_argument("--filter", action="store_true", help="Filter triangles based on the angle")
    parser.add_argument("--scale", type=float, default=0.35, help="Scale of the surface")
    parser.add_argument("--tolerance", type=int, default=20, help="Tolerance angle in degrees")
    parser.add_argument("--surf-normal", action="store_true", help="Plot the surface normal")
    parser.add_argument("--alt-view", action="store_true", help="Alternate view of the surface")
    parser.add_argument("--name", type=str, default="gaf-capture.pdf", help="Name of the output file")

    args = parser.parse_args()

    sns.set_theme(style="whitegrid", color_codes=True)

    extra_args = {}

    if args.surf_normal:
        extra_args = {"axes_labels": ['', '', r'$\hat{n}$'], "axes": 'z'}
    else:
        extra_args = {"axes": ''}

    if args.alt_view:
        extra_args["elev"] = args.theta + 30
        extra_args["azim"] = args.phi
    else:
        extra_args["elev"] = 30

    fig, ax = hemi_coord_figure(hc='c', ha=0.08, zlen=0.3, fontsize=20, **extra_args)

    plot_shadow_map_generation(ax, theta=args.theta, filter=args.filter, scale=args.scale, phi=args.phi,
                               tolerance=args.tolerance)

    plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=0.0, hspace=0.0)
    if args.gen:
        plt.savefig(args.name)
    else:
        plt.show()
