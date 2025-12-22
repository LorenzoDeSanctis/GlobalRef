import numpy as np


def load_dataset(filename):
    """
    Load the Global Reference dataset from a NumPy .npz file.

    Parameters
    ----------
    filename : str
        Path to the dataset file.

    Returns
    -------
    data : dict
        Dictionary containing all dataset arrays and metadata:
        - 'checkerboard_model' : ndarray
        - 'video_stereo_trajectories' : ndarray
        - 'video_monocular_trajectories' : ndarray
        - 'video_fs' : float
        - 'marker_static_recording' : ndarray
        - 'marker_trajectories' : ndarray
        - 'marker_fs' : float
    """
    npz = np.load(filename, allow_pickle=True)

    data = {
        "checkerboard_model": npz["checkerboard_model"],
        "video_stereo_trajectories": npz["video_stereo_trajectories"],
        "video_monocular_trajectories": npz["video_monocular_trajectories"],
        "video_fs": float(npz["video_fs"]),
        "marker_static_recording": npz["marker_static_recording"],
        "marker_trajectories": npz["marker_trajectories"],
        "marker_fs": float(npz["marker_fs"]),
    }

    npz.close()
    return data


def print_dataset_summary(data):
    """
    Print a formatted summary of the loaded dataset.

    Parameters
    ----------
    data : dict
        Dataset dictionary returned by `load_dataset`.
    """
    print("Loaded dataset summary:\n")
    print(f"{'Variable':30s} {'Shape':20s} {'Type':15s}")
    print("-" * 70)

    for key, value in data.items():
        shape = value.shape if hasattr(value, "shape") else "-"
        print(f"{key:30s} {str(shape):20s} {type(value).__name__:15s}")