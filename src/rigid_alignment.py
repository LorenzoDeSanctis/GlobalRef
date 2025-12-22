import numpy as np


def soderkvist(X, Y):
    """
    Estimate the optimal rigid transformation aligning two 3D point sets.

    The transformation is computed in a least-squares sense following
    the Söderkvist and Wedin formulation.

    Parameters
    ----------
    X : ndarray of shape (3, N)
        Reference point set.
    Y : ndarray of shape (3, N)
        Target point set.

    Returns
    -------
    R : ndarray of shape (3, 3)
        Estimated rotation matrix.
    t : ndarray of shape (3, 1)
        Estimated translation vector.
    """
    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)

    Xc = X - mu_x
    Yc = Y - mu_y

    U, _, Vt = np.linalg.svd(Yc @ Xc.T / X.shape[1])

    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    t = mu_y - R @ mu_x

    return R, t


def optimal_tracking(X_flat, Y_flat):
    """
    Denoise a dynamic point set by enforcing rigidity with a static reference.

    The function estimates the rigid transformation that best aligns the
    reference points to the measured points and applies it to the reference.

    Parameters
    ----------
    X_flat : array-like of shape (N, 3) or (3N,)
        Static reference point set.
    Y_flat : array-like of shape (N, 3) or (3N,)
        Measured point set.

    Returns
    -------
    Y_denoised : ndarray of shape (N, 3)
        Rigidly aligned estimate of the measured points.
    """
    X = np.asarray(X_flat).reshape(-1, 3).T
    Y = np.asarray(Y_flat).reshape(-1, 3).T

    R, t = soderkvist(X, Y)
    return (R @ X + t).T
