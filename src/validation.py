import numpy as np
import pytransform3d.transformations as pt
import cv2

from src.synchronization import shift_poses

def rotation_translation_error(R_est, t_est, R_true, t_true):
    """
    Compute rotation and translation errors between an estimated and a ground-truth transform.

    The rotation error is computed as the geodesic distance on SO(3),
    expressed in degrees. The translation error is the Euclidean norm
    of the difference between translation vectors.

    Parameters
    ----------
    R_est : ndarray of shape (3, 3)
        Estimated rotation matrix.
    t_est : ndarray of shape (3,)
        Estimated translation vector.
    R_true : ndarray of shape (3, 3)
        Ground-truth rotation matrix.
    t_true : ndarray of shape (3,)
        Ground-truth translation vector.

    Returns
    -------
    rot_err_deg : float
        Rotation error in degrees.
    trans_err : float
        Translation error (same units as t_true and t_est).
    """
    from src.synchronization import rot_angle

    rot_err_rad = rot_angle(R_true.T @ R_est)
    rot_err_deg = np.degrees(rot_err_rad)

    trans_err = np.linalg.norm(t_true - t_est)

    return rot_err_deg, trans_err


def validate_estimates(R_list, t_list, R_true, t_true):
    """
    Validate multiple estimated transformations against a known ground truth.

    Parameters
    ----------
    R_list : list of ndarray of shape (3, 3)
        List of estimated rotation matrices.
    t_list : list of ndarray of shape (3,)
        List of estimated translation vectors.
    R_true : ndarray of shape (3, 3)
        Ground-truth rotation matrix.
    t_true : ndarray of shape (3,)
        Ground-truth translation vector.

    Returns
    -------
    rot_errors : ndarray
        Rotation errors in degrees.
    trans_errors : ndarray
        Translation errors.
    """
    rot_errors = []
    trans_errors = []

    for R_est, t_est in zip(R_list, t_list):
        r_err, t_err = rotation_translation_error(
            R_est, t_est, R_true, t_true
        )
        rot_errors.append(r_err)
        trans_errors.append(t_err)

    return np.array(rot_errors), np.array(trans_errors)


def orientation_accuracy_RXRY(RA, RB, RX, RY):
    """
    Compute the orientation accuracy metric proposed by Shah et al.

    The metric evaluates the rotational consistency imposed by the
    hand-eye constraint:
        RA * RX ≈ RY * RB

    Parameters
    ----------
    RA : ndarray of shape (3, 3)
        Rotation matrix from pose sequence A.
    RB : ndarray of shape (3, 3)
        Rotation matrix from pose sequence B.
    RX : ndarray of shape (3, 3)
        Rotation matrix of the estimated hand-eye transform X.
    RY : ndarray of shape (3, 3)
        Rotation matrix of the estimated hand-eye transform Y.

    Returns
    -------
    acc : float
        Orientation accuracy in the range [0, 1], where 1 indicates
        perfect consistency.
    """
    diff = RA @ RX - RY @ RB
    diff_norm_sq = np.linalg.norm(diff, ord="fro") ** 2
    acc = 1.0 - diff_norm_sq / 8.0
    return np.clip(acc, 0.0, 1.0)


def position_accuracy_TXTY(TA, TB, X, Y):
    """
    Compute the position accuracy metric proposed by Shah et al.

    The metric evaluates the translational consistency induced by the
    hand-eye transformations.

    Parameters
    ----------
    TA : ndarray of shape (4, 4)
        Pose from sequence A.
    TB : ndarray of shape (4, 4)
        Pose from sequence B.
    X : ndarray of shape (4, 4)
        Estimated hand-eye transformation associated with poses A.
    Y : ndarray of shape (4, 4)
        Estimated hand-eye transformation associated with poses B.

    Returns
    -------
    acc : float
        Position accuracy in the range [0, 1], where 1 indicates
        perfect alignment.
    """
    RA, tA = TA[:3, :3], TA[:3, 3]
    _, tB = TB[:3, :3], TB[:3, 3]

    _, tX = X[:3, :3], X[:3, 3]
    RY, tY = Y[:3, :3], Y[:3, 3]

    pA = RA @ tX + tA
    pB = RY @ tB + tY

    num = np.dot(pA, pB)
    den = (np.linalg.norm(pA) * np.linalg.norm(pB)) + 1e-12

    acc = abs(num / den)
    return np.clip(acc, 0.0, 1.0)


def evaluate_alignment_metrics(poses_A, poses_B, X, Y):
    """
    Evaluate Shah orientation and position accuracy over synchronized pose sequences.

    Parameters
    ----------
    poses_A : list or ndarray of shape (T, 4, 4)
        First synchronized pose sequence.
    poses_B : list or ndarray of shape (T, 4, 4)
        Second synchronized pose sequence.
    X : ndarray of shape (4, 4)
        Estimated hand-eye transformation for poses_A.
    Y : ndarray of shape (4, 4)
        Estimated hand-eye transformation for poses_B.

    Returns
    -------
    ori_acc : ndarray of shape (T,)
        Orientation accuracy values over time.
    pos_acc : ndarray of shape (T,)
        Position accuracy values over time.
    """
    N = len(poses_A)
    ori_acc = np.zeros(N)
    pos_acc = np.zeros(N)

    RX = X[:3, :3]
    RY = Y[:3, :3]

    for i in range(N):
        TA = poses_A[i]
        TB = poses_B[i]

        RA = TA[:3, :3]
        RB = TB[:3, :3]

        ori_acc[i] = orientation_accuracy_RXRY(RA, RB, RX, RY)
        pos_acc[i] = position_accuracy_TXTY(TA, TB, X, Y)

    return ori_acc, pos_acc


def sweep_tau_rigid_error(
    poses_video,
    poses_marker,
    fs,
    taus,
    R_true,
    t_true,
):
    """
    Evaluate rigid-alignment rotation and translation error as a function of
    temporal offset τ.

    For each τ, the pose sequences are synchronized, a hand–eye calibration
    is performed, and the resulting transformation is compared against the
    known ground truth.

    Parameters
    ----------
    poses_video : list or ndarray of shape (T, 4, 4)
        Video-based pose sequence.
    poses_marker : list or ndarray of shape (T, 4, 4)
        Marker-based pose sequence.
    fs : float
        Sampling frequency [Hz].
    taus : ndarray of shape (N,)
        Array of temporal offsets to evaluate [s].
    R_true : ndarray of shape (3, 3)
        Ground-truth rotation matrix.
    t_true : ndarray of shape (3,)
        Ground-truth translation vector.

    Returns
    -------
    rot_err : ndarray of shape (N,)
        Rotation error (degrees) for each τ.
    trans_err : ndarray of shape (N,)
        Translation error for each τ.
    tau_star : float
        Value of τ minimizing a combined rotation–translation error.
    """
    rot_err = np.zeros(len(taus))
    trans_err = np.zeros(len(taus))

    for i, tau in enumerate(taus):
        A_sync, B_sync, _, _ = shift_poses(
            poses_video, poses_marker, fs, tau
        )

        A_R = [T[:3, :3] for T in A_sync]
        A_t = [T[:3, 3] for T in A_sync]
        B_R = [T[:3, :3] for T in B_sync]
        B_t = [T[:3, 3] for T in B_sync]

        X_R, X_t, _, _ = cv2.calibrateRobotWorldHandEye(
            A_R, A_t, B_R, B_t,
            method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
        )

        angle_err, trans_err_val = rotation_translation_error(
            X_R, X_t.flatten(), R_true, t_true
        )

        rot_err[i] = angle_err
        trans_err[i] = trans_err_val

    idx_star = np.argmin(rot_err + 0.1 * trans_err)
    tau_star = taus[idx_star]

    return rot_err, trans_err, tau_star


def sweep_tau_shah_metrics(
    poses_video,
    poses_marker,
    fs,
    taus,
):
    """
    Evaluate Shah orientation and position accuracy metrics as a function of τ.

    For each temporal offset τ, the pose sequences are synchronized, hand–eye
    calibration is performed, and mean ± std of the accuracy metrics are
    computed over the full trajectory.

    Parameters
    ----------
    poses_video : list or ndarray of shape (T, 4, 4)
        Video-based pose sequence.
    poses_marker : list or ndarray of shape (T, 4, 4)
        Marker-based pose sequence.
    fs : float
        Sampling frequency [Hz].
    taus : ndarray of shape (N,)
        Temporal offsets to evaluate [s].

    Returns
    -------
    ori_mean : ndarray of shape (N,)
        Mean orientation accuracy for each τ.
    ori_std : ndarray of shape (N,)
        Standard deviation of orientation accuracy.
    pos_mean : ndarray of shape (N,)
        Mean position accuracy for each τ.
    pos_std : ndarray of shape (N,)
        Standard deviation of position accuracy.
    """
    ori_mean = np.zeros(len(taus))
    ori_std = np.zeros(len(taus))
    pos_mean = np.zeros(len(taus))
    pos_std = np.zeros(len(taus))

    for i, tau in enumerate(taus):
        A_sync, B_sync, _, _ = shift_poses(
            poses_video, poses_marker, fs, tau
        )

        A_R = [T[:3, :3] for T in A_sync]
        A_t = [T[:3, 3] for T in A_sync]
        B_R = [T[:3, :3] for T in B_sync]
        B_t = [T[:3, 3] for T in B_sync]

        X_R, X_t, Y_R, Y_t = cv2.calibrateRobotWorldHandEye(
            A_R, A_t, B_R, B_t,
            method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
        )

        X = pt.transform_from(X_R, X_t.flatten())
        Y = pt.transform_from(Y_R, Y_t.flatten())

        ori_acc, pos_acc = evaluate_alignment_metrics(
            A_sync, B_sync, X, Y
        )

        ori_mean[i] = np.mean(ori_acc)
        ori_std[i] = np.std(ori_acc)
        pos_mean[i] = np.mean(pos_acc)
        pos_std[i] = np.std(pos_acc)

    return ori_mean, ori_std, pos_mean, pos_std