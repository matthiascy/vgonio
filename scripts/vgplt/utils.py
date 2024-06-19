import numpy as np


def rotate(t, axis='x'):
    """Compute the rotation matrix for a given theta and axis."""
    cos = np.cos(t)
    sin = np.sin(t)

    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, cos, -sin],
                         [0, sin, cos]])
    elif axis == 'y':
        return np.array([[cos, 0, sin],
                         [0, 1, 0],
                         [-sin, 0, cos]])
    elif axis == 'z':
        return np.array([[cos, -sin, 0],
                         [sin, cos, 0],
                         [0, 0, 1]])
    else:
        raise ValueError("Invalid axis. Must be 'x', 'y', or 'z'.")


def path_patch_2d_to_3d(pathpatch, m=np.eye(3), z=0):
    """An extension of the pathpatch_2d_to_3d function in matplotlib.

    Convert a 2D Patch to a 3D patch using the given z level and transformation matrix.
    """
    import mpl_toolkits.mplot3d.art3d as art3d
    path = pathpatch.get_path()
    trans = pathpatch.get_patch_transform()

    mpath = trans.transform_path(path)
    pathpatch.__class__ = art3d.PathPatch3D
    pathpatch.set_3d_properties(mpath, z, zdir='z')

    pathpatch._code3d = mpath.codes
    pathpatch._segment3d = np.array([np.matmul(m, (x, y, z)) for x, y in mpath._vertices])


def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates to Cartesian coordinates."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def points_on_hemisphere(r=1.0, ntheta=10, nphi=20):
    """
    Generate a set of points with theta in [0, pi/2] and phi in [0, 2pi].

    :param r: radius of the hemisphere
    :param ntheta: number of points in theta direction (polar angle, from top to bottom)
    :param nphi: number of points in phi direction (azimuthal angle)
    :return: x, y, z, ts, ps

    ts and ps are the theta and phi values of the points, respectively.
    They both have the shape (ntheta, nphi), where ts[:, 0] are the theta values for the first phi value, and
    ps[0, :] are the phi values for the first theta value.

    points with the same theta - xs[1, :], ys[1, :], zs[1, :]
    points with the same phi - xs[:, 1], ys[:, 1], zs[:, 1]
    """
    ts, ps = np.mgrid[0.0:np.pi / 2:ntheta * 1j, 0.0:2.0 * np.pi:nphi * 1j]
    x = r * np.sin(ts) * np.cos(ps)
    y = r * np.sin(ts) * np.sin(ps)
    z = r * np.cos(ts)
    return x, y, z, ts, ps
