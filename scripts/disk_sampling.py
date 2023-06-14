import math
import random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
POINT_SIZE = 2


def rotate_3d(xs, ys, zs, theta, phi):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    theta_rotation = np.array([[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]])
    phi_rotation = np.array([[cos_phi, -sin_phi, 0], [sin_phi, cos_theta, 0], [0, 0, 1]])
    rotation = np.matmul(theta_rotation, phi_rotation)
    return np.matmul(rotation, np.array([xs, ys, zs]))


def uniform_disk_samples(count):
    xs = np.zeros(count)
    ys = np.zeros(count)
    for i in range(count):
        r = math.sqrt(random.uniform(0, 1))
        a = random.uniform(0, math.tau)
        xs[i] = r * math.cos(a)
        ys[i] = r * math.sin(a)
    return xs, ys


def non_uniform_disk_samples(count):
    xs = np.zeros(count)
    ys = np.zeros(count)
    for i in range(count):
        r = random.uniform(0, 1)
        a = random.uniform(0, 2.0 * 3.141592653589793)
        xs[i] = r * math.cos(a)
        ys[i] = r * math.sin(a)
    return xs, ys


def sampling_2d():
    fig, axs = plt.subplots(2, 4)
    fig.suptitle('Disk Sampling')
    axs[0, 0].axis('equal')
    axs[1, 0].axis('equal')

    xs_u, ys_u = uniform_disk_samples(2048)
    axs[0, 0].set_title('uniform disk')
    axs[0, 0].set_xlim([-1.6, 1.6])
    sns.scatterplot(x=xs_u, y=ys_u, ax=axs[0, 0], s=POINT_SIZE)

    for i in range(1, 4):
        ax = axs[0, i]
        ax.axis('equal')
        factor = 1 + i * 0.2
        ax.set_title(f'ellipse, uniform, {factor}x')
        ys_eu = ys_u * factor
        sns.scatterplot(x=xs_u, y=ys_eu, ax=ax, s=POINT_SIZE)

    xs_n, ys_n = non_uniform_disk_samples(2048)
    axs[1, 0].set_title('non-uniform disk')
    axs[1, 0].set_xlim([-1.6, 1.6])
    sns.scatterplot(x=xs_n, y=ys_n, ax=axs[1, 0], s=POINT_SIZE)

    for i in range(1, 4):
        ax = axs[1, i]
        ax.axis('equal')
        factor = 1 + i * 0.2
        ax.set_title(f'ellipse, non-uniform, {factor}x')
        ys_en = ys_n * factor
        sns.scatterplot(x=xs_n, y=ys_en, ax=ax, s=POINT_SIZE)


def sampling_3d():
    xs, ys = uniform_disk_samples(1024)
    zs = np.repeat(2.0, 1024)
    fig, axs = plt.subplots(3, 3, subplot_kw={'projection': '3d'})
    fig.suptitle('Sampling')
    xs = xs * 3.0
    ys = ys * 3.0

    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            ax = axs[i, j]
            ax.set_xlim([-4, 4])
            ax.set_ylim([-4, 4])
            ax.set_zlim([-4, 4])
            theta = idx * math.pi / 18
            factor = 1 / math.cos(theta)
            ax.set_title(f'θ = {math.degrees(theta):.0f}°, factor = {factor:.2}')
            ys_f = ys * factor
            rotated = rotate_3d(xs, ys_f, zs, theta, 0)
            ax.scatter3D(xs=rotated[0], ys=rotated[1], zs=rotated[2], s=0.5)


if __name__ == "__main__":
    sns.set(style="whitegrid")

    sampling_2d()
    sampling_3d()

    plt.show()
