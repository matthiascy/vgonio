import math

import numpy as np
import matplotlib.pyplot as plt


def points_on_hemisphere():
    count_phi = 40
    count_theta = 20
    d_phi = math.tau / count_phi
    d_theta = np.pi * 0.5 / count_theta
    points = np.zeros((count_phi * (count_theta + 1), 3))
    for n_phi in range(0, count_phi):
        for n_theta in range(0, count_theta + 1):
            phi = n_phi * d_phi
            theta = n_theta * d_theta
            x = math.cos(phi) * math.sin(theta)
            y = math.sin(phi) * math.sin(theta)
            z = math.cos(theta)
            points[n_phi * (count_theta + 1) + n_theta, 0] = x
            points[n_phi * (count_theta + 1) + n_theta, 1] = y
            points[n_phi * (count_theta + 1) + n_theta, 2] = z
    return points


def hemisphere2disk_equal_area(pts):
    theta = np.arccos(pts[:, 2])
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    r = np.sin(theta * 0.5) * np.sqrt(2.0)
    xs = r * np.cos(phi)
    ys = r * np.sin(phi)
    return xs, ys


def hemisphere2disk_conformal(pts):
    theta = np.arccos(pts[:, 2])
    phi = np.arctan2(pts[:, 1], pts[:, 0])
    r = np.tan(theta * 0.5)
    xs = r * np.cos(phi)
    ys = r * np.sin(phi)
    return xs, ys


if __name__ == '__main__':
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })

    pts = points_on_hemisphere()
    fig = plt.figure()

    ax = fig.add_subplot(1, 4, 1, projection='3d')
    ax.set_title('Hemisphere')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)

    ax = fig.add_subplot(1, 4, 2)
    ax.set_aspect('equal')
    ax.set_title('Naive')
    ax.scatter(pts[:, 0], pts[:, 1], s=1)

    ax = fig.add_subplot(1, 4, 3)
    ax.set_aspect('equal')
    ax.set_title(r'Equal Area - $\sqrt{2} * \sin\frac{\theta}{2}$')
    xs, ys = hemisphere2disk_equal_area(pts)
    ax.scatter(xs, ys, s=2)

    ax = fig.add_subplot(1, 4, 4)
    ax.set_aspect('equal')
    ax.set_title('Conformal')
    xs, ys = hemisphere2disk_conformal(pts)
    ax.scatter(xs, ys, s=1)

    plt.show()
