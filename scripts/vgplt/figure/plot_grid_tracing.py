import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from mpl_toolkits.mplot3d import art3d


# Function to draw grid tracing using DDA algorithm
def dda_line(x0, y0, x1, y1):
    cells = []
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy))
    x_inc = dx / steps
    y_inc = dy / steps

    x, y = x0, y0
    for _ in range(int(steps) + 1):
        cells.append((int(x), int(y)))
        x += x_inc
        y += y_inc

    return cells


# Function to calculate intersection points
def intersection_points(x0, y0, x1, y1):
    intersections = []
    if x1 != x0:
        m = (y1 - y0) / (x1 - x0)
        for x in range(min(int(x0), int(x1)), max(int(x0), int(x1)) + 1):
            y = m * (x - x0) + y0
            intersections.append((x, y))
    if y1 != y0:
        m_inv = (x1 - x0) / (y1 - y0)
        for y in range(min(int(y0), int(y1)), max(int(y0), int(y1)) + 1):
            x = m_inv * (y - y0) + x0
            intersections.append((x, y))
    return intersections


def plot_grid(ax):
    # Define the grid size
    grid_size = 5

    # Create grid lines
    for i in range(grid_size + 1):
        ax.axhline(i, color='lightgray', linewidth=0.5)
        ax.axvline(i, color='lightgray', linewidth=0.5)

    # Define start and end points of the line
    x0, y0 = -1, -1
    x1, y1 = 3, 2.2
    m = (y1 - y0) / (x1 - x0)

    # Get the cells that the line intersects
    cells = dda_line(x0, y0, x1, y1)

    # Highlight the cells
    for cell in cells[1:]:
        rect = patches.Rectangle((cell[0], cell[1]), 1, 1, edgecolor='none', facecolor='lightcyan', alpha=0.3)
        ax.add_patch(rect)

    # Calculate and plot the intersection points
    intersections = intersection_points(x0, y0, x1, y1)

    highlighted_cells = set((int(x), int(y)) for x, y in intersections)
    for cell in highlighted_cells:
        if cell not in cells:
            rect = patches.Rectangle((cell[0], cell[1]), 1, 1, edgecolor='none', facecolor='honeydew', alpha=0.3)
            ax.add_patch(rect)

    # Calculate the slope of the line then the end point
    endpoint = (x1 + 0.4, y1 + m * 0.4)
    ax.plot([x0, endpoint[0]], [y0, endpoint[1]], color='k', linewidth=2, marker='o', markersize=5)
    ax.plot([endpoint[0], endpoint[0] + 0.7], [endpoint[1], endpoint[1] + 0.7 * m], color='k', linewidth=2,
            linestyle='--')
    arrow = patches.FancyArrowPatch((endpoint[0] + 0.7, endpoint[1] + 0.7 * m),
                                    (endpoint[0] + 0.9, endpoint[1] + 0.9 * m), color='k',
                                    arrowstyle='->', mutation_scale=15, linewidth=2)
    ax.add_patch(arrow)

    for point in intersections[1:]:
        ax.plot(point[0], point[1], color='tomato', marker='o')

    # Set the axis limits
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 1.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Remove axis labels but keep ticks
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size - 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def ray_tri_intersect(p0, p1, p2, org, dir):
    # Calculate the normal of the triangle formed by the corners 0, 2, and 3
    n = np.cross(p1 - p0, p2 - p0)  # Calculate the normal of the triangle
    d = -np.dot(n, p0)  # Calculate the distance from the origin to the plane
    t = -(np.dot(n, org) + d) / np.dot(n, dir)  # Calculate the intersection point
    return org + t * dir, n


def reflect(v, n):
    return v - 2 * np.dot(v, n) * n


def plot_cell(ax, elev=30, azim=-35, tri=False, alt=False, walls="", rfl=True, ray=False, fontsize=15):
    # Set up the plot
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    ax.set_aspect("equal")
    ax.set_proj_type('ortho')
    ax.grid(False)
    # hide xz plane and yz plane
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0), alpha=0.0)
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0), alpha=0.0)
    # Draw subtle transparent planes to represent the xz and yz planes
    xs, ys = np.meshgrid([0, 1], [0, 1])

    max_val = 0.65

    if 'l' in walls:
        # left wall
        ax.plot_surface(xs, np.zeros_like(xs), ys * max_val, color='c', alpha=0.05, linewidth=0)
    if 'b' in walls:
        # back wall
        ax.plot_surface(np.zeros_like(xs), ys, xs * max_val, color='c', alpha=0.05, linewidth=0)
    if 'r' in walls:
        # right wall
        ax.plot_surface(xs, np.ones_like(ys), ys * max_val, color='c', alpha=0.05, linewidth=0)
    if 'f' in walls:
        # front wall
        ax.plot_surface(np.ones_like(xs), ys, xs * max_val, color='c', alpha=0.05, linewidth=0)

    # hide axis line
    ax.xaxis.line.set_color("white")
    ax.yaxis.line.set_color("white")
    ax.zaxis.line.set_color("white")
    # hide ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)

    # Plot 4 corner points
    corners = np.array([[0, 0, 0.2], [1, 0, 0.50], [1, 1, 0.4], [0, 1, max_val]])
    ax.scatter(*corners.T, color='b', s=60, linewidths=1)
    # Plot the line from each corner to the ground
    for i, corner in enumerate(corners):
        ax.plot([corner[0], corner[0]], [corner[1], corner[1]], [0, corner[2]], color='b', linestyle='--', linewidth=2)
        delta = 0.08 if i != 0 else 0.03
        ax.text(corner[0], corner[1], corner[2] + delta, fr'$c_{i}$', color='k', fontsize=fontsize)

    if tri:
        if alt:
            # Plot with different order of corners to show another way of triangulating the cell
            ax.plot_trisurf(corners[[1, 2, 3, 0], 0], corners[[1, 2, 3, 0], 1], corners[[1, 2, 3, 0], 2],
                            color='goldenrod',
                            alpha=0.1)
        else:
            # Connect the corners to form two triangles and plot the surface
            ax.plot_trisurf(corners[:, 0], corners[:, 1], corners[:, 2], color='goldenrod', alpha=0.1)

    if ray:
        # Draw a line on the ground
        m = (1.0 + 0.3) / (0.5 - 0.1)
        ax.plot([0.1, 0.5], [-0.3, 1.0], [0, 0], color='k', linestyle='--', alpha=0.6)
        # Add arrow at the end of the line
        arrow = patches.FancyArrowPatch((0.5, 1.0), (0.575, 1.0 + m * 0.075), color='k', arrowstyle='->',
                                        mutation_scale=400,
                                        linewidth=3, fill=False, alpha=0.6)
        ax.add_patch(arrow)
        art3d.pathpatch_2d_to_3d(arrow, z=0, zdir='z')

        # Draw a line in the air with the same x and y coordinates as the ground line
        start = np.array([0.1, -0.3, 0.5])
        end = np.array([0.5, 1.0, 0.15])
        dir = (end - start) / np.linalg.norm(end - start)
        enter = start + (0.3 / dir[1]) * dir
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color='k', linestyle='--')
        ax.quiver(*end, *dir, color='k', arrow_length_ratio=0.2, linewidth=3, length=0.26)

        if rfl:
            # Calculate the intersection points of the line in the air with triangle
            p, n = ray_tri_intersect(corners[0], corners[2], corners[3], start, dir) if alt else ray_tri_intersect(
                corners[0],
                corners[1],
                corners[3],
                start, dir)
            ax.scatter(*p, color='r', s=50, linewidths=0)
            r = reflect(dir, n)
            ax.quiver(*p, *r, color='r', arrow_length_ratio=0.15, linewidth=3, length=0.3, zorder=10)
            # Draw the normal of the triangle in dashed line
            ax.quiver(*p, *n, color='c', arrow_length_ratio=0.05, linewidth=3, length=0.2, linestyle='--', zorder=10)
            # Annotate the normal
            ax.text(p[0] - 0.05, p[1] - 0.05, p[2] + 0.2, r'$\mathbf{n}$', color='c', fontsize=fontsize)

        # Draw the point where the line in the air intersects the left wall
        ax.scatter(*enter, color='r', s=50, linewidths=0)
        ax.text(enter[0], enter[1], enter[2] + 0.06, r'$P_a$', color='k', fontsize=fontsize)
        # Draw the line from the intersection point to the ground
        ax.plot([enter[0], enter[0]], [enter[1], enter[1]], [0, enter[2]], color='r', linestyle='--')

        # Draw the point where the line in the air intersects the right wall
        ax.scatter(*end, color='r', s=50, linewidths=0)
        ax.text(end[0], end[1], end[2] + 0.05, r'$P_b$', color='k', fontsize=fontsize)
        # Draw the line from the intersection point to the ground
        ax.plot([end[0], end[0]], [end[1], end[1]], [0, end[2]], color='r', linestyle='--')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", action="store_true", help="Generate figures")
    parser.add_argument("--grid", action="store_true", help="Generate overview figure")
    parser.add_argument("--cell", action="store_true", help="Generate cell figure")
    parser.add_argument("--tri", action="store_true", help="Triangulate the cell")
    parser.add_argument("--alt", action="store_true", help="Alternative triangulation")
    parser.add_argument("--fontsize", type=int, default=20, help="Font size for labels")
    parser.add_argument("--ray", action="store_true", help="Show ray intersection")
    parser.add_argument("--rfl", action="store_true", help="Show reflection")
    parser.add_argument("--walls", help="Walls to show", default="")
    args = parser.parse_args()

    sns.set(style="whitegrid", color_codes=True)

    kwargs = {
        'tri': args.tri,
        'alt': args.alt,
        'fontsize': args.fontsize,
        'ray': args.ray,
        'rfl': args.rfl,
        'walls': args.walls
    }

    if args.grid:
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111)
        plot_grid(ax)
        plt.tight_layout()
        if args.gen:
            plt.savefig("grid-tracing-grid.pdf")
        else:
            plt.show()
    elif args.cell:
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        plot_cell(ax, **kwargs)

        plt.tight_layout()
        if args.gen:
            plt.savefig("grid-tracing-cell.pdf")
        else:
            plt.show()
