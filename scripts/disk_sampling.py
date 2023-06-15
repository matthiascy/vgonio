import math
import random
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
POINT_SIZE = 2


def rotate_x(xs, ys, zs, theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rotation = np.array([[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]])
    return np.matmul(rotation, np.array([xs, ys, zs]))


def rotate_y(xs, ys, zs, theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rotation = np.array([[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]])
    return np.matmul(rotation, np.array([xs, ys, zs]))


def rotate_z(xs, ys, zs, theta):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rotation = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])
    return np.matmul(rotation, np.array([xs, ys, zs]))


def rotate_3d(xs, ys, zs, theta, phi):
    rotated = rotate_y(xs, ys, zs, theta)
    rotated = rotate_z(rotated[0], rotated[1], rotated[2], phi)
    return rotated


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
    axs[0, 0].set_xlim([-1.0, 1.0])
    axs[0, 0].set_ylim([-1.0, 1.0])
    sns.scatterplot(x=xs_u, y=ys_u, ax=axs[0, 0], s=POINT_SIZE)

    for i in range(1, 4):
        ax = axs[0, i]
        ax.axis('equal')
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([-1.0, 1.0])
        factor = 1 - i * 0.15
        ax.set_title(f'ellipse, uniform, {factor}x')
        ys_eu = ys_u * factor
        sns.scatterplot(x=xs_u, y=ys_eu, ax=ax, s=POINT_SIZE)

    xs_n, ys_n = non_uniform_disk_samples(2048)
    axs[1, 0].set_title('non-uniform disk')
    axs[1, 0].set_xlim([-1.0, 1.0])
    axs[1, 0].set_ylim([-1.0, 1.0])
    sns.scatterplot(x=xs_n, y=ys_n, ax=axs[1, 0], s=POINT_SIZE)

    for i in range(1, 4):
        ax = axs[1, i]
        ax.axis('equal')
        factor = 1 - i * 0.15
        ax.set_title(f'ellipse, non-uniform, {factor}x')
        ys_en = ys_n * factor
        sns.scatterplot(x=xs_n, y=ys_en, ax=ax, s=POINT_SIZE)


def sampling_3d(phi):
    xs, ys = uniform_disk_samples(1024)
    zs = np.repeat(1.0, 1024)
    fig, axs = plt.subplots(3, 3, subplot_kw={'projection': '3d'})
    fig.suptitle(f'Sampling φ = {math.degrees(phi):.2f}°')

    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            ax = axs[i, j]
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            theta = idx * math.pi / 18
            factor = math.cos(theta)
            ax.set_title(f'θ = {math.degrees(theta):.0f}°, factor = {factor:.2}')
            xs_f = xs * factor
            final = rotate_3d(xs_f, ys, zs, theta, phi)
            ax.scatter3D(xs=final[0], ys=final[1], zs=final[2], s=0.5)


def sampling_2d_v2():
    xs, ys = uniform_disk_samples(1024)
    zs = np.repeat(0.0, 1024)
    fig, axs = plt.subplots(6, 6)
    fig.suptitle('Rotate around Z')

    for i in range(6):
        for j in range(6):
            idx = i * 6 + j
            ax = axs[i, j]
            ax.axis('equal')
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            theta = idx * math.pi / 18
            ax.set_title(f'θ = {math.degrees(theta):.0f}°')
            rotated = rotate_z(xs * math.cos(theta), ys, zs, theta)
            sns.scatterplot(x=rotated[0], y=rotated[1], ax=ax, s=1)


if __name__ == "__main__":
    sns.set(style="whitegrid")

    sampling_2d_v2()

    sampling_2d()

    for i in range(6):
        sampling_3d(i * math.pi / 3)

    plt.show()
