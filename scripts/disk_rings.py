import math
import matplotlib.pyplot as plt
import numpy as np


def compute_ring_radius(count, index, radius):
    """Compute the radius of a ring of the same area on a disk.

    r_0 is always 0, and r_N is always the radius of the disk.

    pi * (r_i ** 2 - r_{i-1} ** 2) = pi * r_disk ** 2 / count

    r_i = r_disk * sqrt(i / count)

    count: number of rings
    index: index of the ring
    radius: radius of the disk
    """
    return radius * math.sqrt(index / count)


def plot_disk_rings(count, radius, ax):
    """Plot the rings of a disk.

    count: number of rings
    radius: radius of the disk
    """
    ax.set_aspect("equal")
    angles = np.linspace(0, 2 * math.pi, 64)
    xs = np.cos(angles)
    ys = np.sin(angles)

    ax.plot(0, 0, color='red', marker='x')
    ax.annotate(r"$r_0$", xy=(0+0.05, 0+0.05))

    for i in range(1, count+1):
        r_internal = compute_ring_radius(count, i - 1, radius)
        r_i = compute_ring_radius(count, i, radius)
        ax.plot(r_i * xs, r_i * ys, color="green")
        ax.plot(r_i, 0, color="red", marker="x")
        ax.annotate(r"$r_{}$".format(i), xy=(r_i+0.025, 0+0.025))
        ax.plot([r_internal, r_i], [0, 0], color="blue")
        ax.set_title(r"Equal-area rings - $r_i=R\sqrt{\frac{i}{N}}$")


def becker_compute_ks(k0, N):
    ks = np.zeros(N)
    ks[0] = k0
    for i in range(1, N):
        ks[i] = np.round(np.square(np.sqrt(ks[i-1]) + np.sqrt(np.pi)))
    return ks


def becker_compute_rs(ks, N, R=1.0):
    rs = np.zeros(N)
    rs[0] = R * np.sqrt(ks[0] / ks[N-1])
    for i in range(1, N):
        rs[i] = np.sqrt(ks[i] / ks[i-1]) * rs[i-1]
    return rs


def becker_plot_rings(ks, rs, ax):
    ax.set_title("Becker's method")
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    angles = np.linspace(0, 2 * math.pi, 64)
    xs = np.cos(angles)
    ys = np.sin(angles)

    for (i, (r, k)) in enumerate(zip(rs, ks)):
        ax.plot(r * xs, r * ys, color="green")
        if k > 1:
            k_prev = ks[i-1] if i - 1 >= 0 else 0
            dk = k - k_prev
            theta = np.pi * 2.0 / dk
            for j in range(int(dk)):
                r_prev = rs[i-1] if i - 1 >= 0 else 0
                x0 = np.cos(theta * j)
                y0 = np.sin(theta * j)
                ax.plot([r_prev * x0, r * x0], [r_prev * y0, r * y0], color="blue")


def becker_compute_theta(ks, rs, N):
    # thetas = np.zeros(N)
    # thetas[0] = np.arccos(rs[0] * rs[0] / 2.0 - 1.0)
    # for i in range(1, N):
    #     thetas[i] = thetas[i-1] - 2.0 * np.sin(thetas[i-1] / 2.0) * np.sqrt(np.pi / ks[i-1])
    # return thetas
    return 2.0 * np.arcsin(rs / 2.0)


def becker_plot_hemisphere(ks, rs, ts, N, ax):
    # ax.set_aspect("equal")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(0.0, 2.0)
    angles = np.linspace(0, 2 * math.pi, 128)

    for (i, (t, k)) in enumerate(zip(ts, ks)):
        zs = np.cos(t)
        r = np.sin(t)
        xs = np.cos(angles) * r
        ys = np.sin(angles) * r
        ax.plot(xs, ys, zs, color="green")
        if k > 1:
            k_prev = ks[i-1] if i - 1 >= 0 else 0
            n = k - k_prev
            t_prev = ts[i-1] if i - 1 >= 0 else 0
            for j in range(int(n)):
                r_prev = np.sin(t_prev)
                f = j * 2.0 * np.pi / n
                xs = np.cos(f) * np.array([r_prev, r])
                ys = np.sin(f) * np.array([r_prev, r])
                zs = np.cos(np.array([t_prev, t]))
                ax.plot(xs, ys, zs, color="blue")


if __name__ == "__main__":
    figure = plt.figure()
    # plot_disk_rings(8, 1, figure.add_subplot(1, 3, 1))

    N = 10
    R = np.sqrt(2.0)
    ks = becker_compute_ks(1, N)
    rs = becker_compute_rs(ks, N, R)
    ts = becker_compute_theta(ks, rs, N)
    print("ks: ", ks)
    print("rs: ", rs)
    print("dr: ", rs[1:] - rs[:-1])
    print("ns: ", ks[1:] - ks[:-1])
    print("ts: ", ts)
    becker_plot_rings(ks, rs, figure.add_subplot(1, 2, 1))
    becker_plot_hemisphere(ks, rs, ts, N, figure.add_subplot(1, 2, 2, projection="3d"))

    plt.show()
