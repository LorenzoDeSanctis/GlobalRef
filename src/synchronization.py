import numpy as np
import pytransform3d.transformations as pt


def relatives(frames):
    """
    Compute relative SE(3) motions between consecutive poses.

    Each relative transform is computed as:
        T_rel[i] = inv(T[i]) @ T[i+1]

    Parameters
    ----------
    frames : list or ndarray of shape (T, 4, 4)
        Sequence of absolute SE(3) poses.

    Returns
    -------
    relatives : list of ndarray of shape (4, 4)
        Relative transformations between consecutive poses.
    """
    return [
        pt.concat(pt.invert_transform(frames[i]), frames[i + 1])
        for i in range(len(frames) - 1)
    ]


def rot_angle(R):
    """
    Compute the rotation angle of a rotation matrix using a robust atan2 formulation.

    The angle is computed from the skew-symmetric part of R and is robust
    near singular configurations.

    Parameters
    ----------
    R : ndarray of shape (3, 3)
        Rotation matrix.

    Returns
    -------
    theta : float
        Rotation angle in radians.
    """
    W = 0.5 * (R - R.T)

    wx = W[2, 1]
    wy = W[0, 2]
    wz = W[1, 0]

    sin_theta = np.linalg.norm([wx, wy, wz])
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)

    return np.arctan2(sin_theta, cos_theta)


def xcorr_shift_subsample(a, b):
    """
    Estimate the sub-sample temporal shift maximizing cross-correlation.

    The signals are normalized and cross-correlated in the frequency domain.
    A parabolic interpolation around the correlation peak is used to refine
    the shift estimate to sub-sample accuracy.

    Parameters
    ----------
    a, b : array-like of shape (N,)
        Input scalar signals.

    Returns
    -------
    lag_int : int
        Integer-valued lag (in samples).
    lag_hat : float
        Refined lag estimate (in samples).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)

    N = 1 << int(np.ceil(np.log2(len(a) + len(b) - 1)))

    A = np.fft.rfft(a, n=N)
    B = np.fft.rfft(b, n=N)

    corr = np.fft.irfft(A * np.conj(B), n=N)
    corr = np.roll(corr, len(b) - 1)

    lags = np.arange(-len(b) + 1, len(a))
    corr_valid = corr[:len(lags)]

    idx = np.argmax(corr_valid)
    lag_int = int(lags[idx])

    if 0 < idx < len(corr_valid) - 1:
        y1 = corr_valid[idx - 1]
        y2 = corr_valid[idx]
        y3 = corr_valid[idx + 1]

        denom = y1 - 2.0 * y2 + y3
        if abs(denom) > 1e-12:
            delta = 0.5 * (y1 - y3) / denom
            lag_hat = lags[idx] + delta
        else:
            lag_hat = float(lags[idx])
    else:
        lag_hat = float(lags[idx])

    return lag_int, lag_hat


def estimate_shift(poses_A, poses_B, fs):
    """
    Estimate the temporal offset between two pose sequences.

    The estimation is based on cross-correlation of the rotation angle
    extracted from relative SE(3) motions.

    Parameters
    ----------
    poses_A : list or ndarray of shape (T, 4, 4)
        First pose sequence.
    poses_B : list or ndarray of shape (T, 4, 4)
        Second pose sequence.
    fs : float
        Sampling frequency [Hz].

    Returns
    -------
    tau_sec : float
        Estimated temporal offset in seconds.
        Positive values indicate that poses_B lags behind poses_A.
    tau_samples : float
        Estimated temporal offset in samples.
    """
    rel_A = relatives(poses_A)
    rel_B = relatives(poses_B)

    a_vals = np.array([rot_angle(T[:3, :3]) for T in rel_A])
    b_vals = np.array([rot_angle(T[:3, :3]) for T in rel_B])

    _, tau_samples = xcorr_shift_subsample(a_vals, b_vals)
    tau_sec = tau_samples / fs

    return tau_sec, tau_samples


def shift_poses(poses_A, poses_B, fs, tau):
    """
    Synchronize two SE(3) pose sequences using temporal shifting and ScLERP.

    The second sequence is shifted by tau seconds and interpolated onto
    the time base of the first sequence.

    Parameters
    ----------
    poses_A : ndarray of shape (T1, 4, 4)
        Reference pose sequence.
    poses_B : ndarray of shape (T2, 4, 4)
        Pose sequence to be shifted.
    fs : float
        Sampling frequency [Hz].
    tau : float
        Temporal offset in seconds.
        Positive values indicate that poses_B lags behind poses_A.

    Returns
    -------
    poses_A_sync : ndarray of shape (N, 4, 4)
        Reference poses on the common timeline.
    poses_B_sync : ndarray of shape (N, 4, 4)
        Time-shifted and interpolated poses.
    fs_common : float
        Common sampling frequency [Hz].
    t_common : ndarray of shape (N,)
        Common time vector.
    """
    poses_A = np.asarray(poses_A)
    poses_B = np.asarray(poses_B)

    t_A = np.arange(len(poses_A)) / fs
    t_B = np.arange(len(poses_B)) / fs
    t_B_shifted = t_B + tau

    t_common = t_A.copy()
    fs_common = fs

    def interpolate_poses(poses, t_src, t_query):
        out = []

        for tq in t_query:
            if tq <= t_src[0]:
                out.append(poses[0])
                continue
            if tq >= t_src[-1]:
                out.append(poses[-1])
                continue

            i = np.searchsorted(t_src, tq) - 1
            i = np.clip(i, 0, len(t_src) - 2)

            t0, t1 = t_src[i], t_src[i + 1]
            alpha = (tq - t0) / (t1 - t0 + 1e-12)

            T0 = poses[i]
            T1 = poses[i + 1]

            out.append(pt.transform_sclerp(T0, T1, alpha))

        return np.array(out)

    poses_B_sync = interpolate_poses(poses_B, t_B_shifted, t_common)
    poses_A_sync = poses_A.copy()

    return poses_A_sync, poses_B_sync, fs_common, t_common
