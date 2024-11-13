import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates


def pixel_coord_to_spherical_coord(px, py, w, h):
    # convert into [-1, 1]
    x = (px + 0.5) / w * 2 - 1
    y = (py + 0.5) / h * 2 - 1
    if x ** 2 + y ** 2 > 1:
        return None
    z = (1 - x ** 2 - y ** 2) ** 0.5
    phi = np.arctan2(y, x)
    theta = np.arccos(z)
    if phi < 0:
        phi += 2 * np.pi

    return theta, phi


def spherical_to_image_coord(t, p, w, h):
    # reverse the conversion
    x = np.sin(t) * np.cos(p)
    y = np.sin(t) * np.sin(p)
    # convert x and y to normalized image coordinates
    px = (x / 2 + 0.5) * w
    py = (y / 2 + 0.5) * h
    print(
        f"Pixel coordinates of the spherical point: {np.degrees(t):.2f} {np.degrees(p):.2f}, is {px} {py}")

    return px, py


def interpolate(data_image, theta, phi, order=3):
    # Get the dimensions of the image
    height, width = data_image.shape

    # Convert spherical coordinates to circular image pixel coordinates
    x_pixel, y_pixel = spherical_to_image_coord(theta, phi, width, height)

    # Check if the point is within the bounds of the unit circle in the image
    if (x_pixel - width / 2) ** 2 + (y_pixel - height / 2) ** 2 > (min(width, height) / 2) ** 2:
        return None  # Point is outside the hemisphere projection

    # Perform bicubic interpolation
    value = map_coordinates(data_image, [[x_pixel], [y_pixel]], order=3)

    return value[0]  # Extract the interpolated value


if __name__ == '__main__':
    # Set grid dimensions
    width, height = 8, 8

    # generate a random image
    data_image = np.random.rand(height, width)

    for i in range(height):
        for j in range(width):
            print(data_image[i, j], end=", ")
            if j == width - 1:
                print()

    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(121)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    # Create visualization grid and compute spherical coordinates for each pixel
    for px in range(width):
        for py in range(height):
            result = pixel_coord_to_spherical_coord(px, py, width, height)
            if result is not None:
                theta, phi = result
                ax.text(px + 0.5, py + 0.5,
                        f"θ={np.degrees(theta):.2f}\nφ={np.degrees(phi):.2f}\n{data_image[px, py]:.2f}",
                        ha='center',
                        va='center', fontsize=8)
            else:
                ax.text(px + 0.5, py + 0.5,
                        f"{data_image[px, py]:.2f}",
                        ha='center',
                        va='center', fontsize=8)
            # Draw pixel as a square
            ax.add_patch(plt.Rectangle((px, py), 1, 1, edgecolor='black', facecolor='lightgrey'))

    # Set labels and show grid
    ax.set_xticks(np.arange(0, width + 1, 1))
    ax.set_yticks(np.arange(0, height + 1, 1))
    ax.invert_yaxis()
    ax.grid(True)

    ax3d = fig.add_subplot(122, projection='3d')
    # Plot each pixel on the sphere
    for px in range(width):
        for py in range(height):
            result = pixel_coord_to_spherical_coord(px, py, width, height)
            if result is not None:
                theta, phi = result
                # Convert spherical coordinates to Cartesian coordinates
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)

                # Plot each point
                ax3d.scatter(x, y, z, color="b", s=20)
                ax3d.text(x, y, z, f"θ={np.degrees(theta):.2f}\nφ={np.degrees(phi):.2f}", fontsize=6, ha="center",
                          color="black")
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title("2D Pixel Coordinates Mapped to a 3D Sphere")
    ax3d.view_init(elev=20, azim=30)

    theta_iterp = np.radians(30)
    phi_iterp = np.radians(210)
    print(
        f"Interpolated(bicubic) value at (θ={np.degrees(theta_iterp):.2f}, φ={np.degrees(phi_iterp):.2f}): {interpolate(data_image, theta_iterp, phi_iterp, 3)}")

    plt.title("2D Pixel to 3D Spherical Coordinates")
    plt.show()
