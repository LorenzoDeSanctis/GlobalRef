import numpy as np
from scipy.interpolate import CubicSpline


def upsample_trajectories(data, fs_in, fs_out):
    """
    Temporally upsample 3D point trajectories using cubic spline interpolation.

    Each coordinate of each point is interpolated independently on the
    original time base and resampled at the target frequency.

    Parameters
    ----------
    data : ndarray of shape (T, N, 3)
        Input trajectories sampled at fs_in.
    fs_in : float
        Original sampling frequency [Hz].
    fs_out : float
        Target sampling frequency [Hz].

    Returns
    -------
    data_up : ndarray of shape (T_out, N, 3)
        Upsampled trajectories.
    t_out : ndarray of shape (T_out,)
        Time vector associated with the upsampled trajectories.
    """
    T, N, D = data.shape

    t_in = np.arange(T) / fs_in
    t_out = np.arange(int(t_in[-1] * fs_out) + 1) / fs_out

    data_up = np.zeros((len(t_out), N, D))

    for j in range(N):
        for d in range(D):
            cs = CubicSpline(t_in, data[:, j, d])
            data_up[:, j, d] = cs(t_out)

    return data_up, t_out
