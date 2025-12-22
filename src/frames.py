import numpy as np


def transform_from_points(p1, p2, p3):
    """
    Construct a right-handed SE(3) frame from three non-collinear 3D points.

    The origin is placed at p1, the x-axis is defined by the direction from
    p1 to p2, and the xy-plane is defined by p3.

    Parameters
    ----------
    p1, p2, p3 : array-like of shape (3,)
        3D coordinates of the reference points.

    Returns
    -------
    T : ndarray of shape (4, 4)
        Homogeneous transformation matrix representing the constructed frame.
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)

    x_axis = p2 - p1
    x_axis /= np.linalg.norm(x_axis)

    v = p3 - p1
    z_axis = np.cross(x_axis, v)
    z_axis /= np.linalg.norm(z_axis)

    y_axis = np.cross(z_axis, x_axis)

    T = np.eye(4)
    T[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    T[:3, 3] = p1

    return T
