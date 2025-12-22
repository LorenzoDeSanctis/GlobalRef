import numpy as np
import cv2
import pytransform3d.transformations as pt


def extract_rt(poses):
    """
    Extract rotation matrices and translation vectors from a sequence of SE(3) poses.

    Parameters
    ----------
    poses : list or ndarray of shape (T, 4, 4)
        Sequence of homogeneous transformation matrices.

    Returns
    -------
    R_list : list of ndarray of shape (3, 3)
        Rotation matrices extracted from the poses.
    t_list : list of ndarray of shape (3,)
        Translation vectors extracted from the poses.
    """
    R_list = [T[:3, :3] for T in poses]
    t_list = [T[:3, 3] for T in poses]
    return R_list, t_list


def estimate_handeye_shah(poses_A, poses_B):
    """
    Estimate the hand-eye / robot-world transformations using the Shah method.

    The function wraps OpenCV's `calibrateRobotWorldHandEye` implementation
    and returns the estimated transformations as SE(3) matrices.

    Parameters
    ----------
    poses_A : list or ndarray of shape (T, 4, 4)
        First synchronized pose sequence (e.g., marker-based).
    poses_B : list or ndarray of shape (T, 4, 4)
        Second synchronized pose sequence (e.g., video-based).

    Returns
    -------
    X : ndarray of shape (4, 4)
        Estimated transformation from frame A to the common world frame.
    Y : ndarray of shape (4, 4)
        Estimated transformation from frame B to the common world frame.
    """
    A_R, A_t = extract_rt(poses_A)
    B_R, B_t = extract_rt(poses_B)

    X_R, X_t, Y_R, Y_t = cv2.calibrateRobotWorldHandEye(
        A_R, A_t,
        B_R, B_t,
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
    )

    X = pt.transform_from(X_R, X_t.flatten())
    Y = pt.transform_from(Y_R, Y_t.flatten())

    return X, Y