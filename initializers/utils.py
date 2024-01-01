import numpy as np


def rt_matrices_2_transform_matrix(r, t):
    """
    Makes a transformation matrix from the given rotation matrix and translation vector
    :param r:  The rotation matrix
    :param t: The translation vector
    :return transform (ndarray): The transformation matrix
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = r
    transform[:3, 3] = t
    return transform
