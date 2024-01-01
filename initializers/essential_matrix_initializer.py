import cv2
import numpy as np
from tqdm import tqdm
from .utils import rt_matrices_2_transform_matrix


# This function should detect & compute keypoint & descriptors from the i-1'th and ith image using the class orb object
# The descriptors should then be matched using the class flann object (knnMatch with k=2)
# Remove the matches not satisfying Lowe's ratio test
# Return a list of the good matches for each image, sorted such that the nth descriptor in image i matches the nth
# descriptor in image i-1
# https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html

class EssentialMatrixInitializer:
    # def __init__(self, save_dir, dataloader, feature, map_data):
    #     self.save_dir = save_dir
    #     self.dataloader = dataloader
    #     self.feature = feature
    #     self.map_data = map_data

    def initialize(self, q1, q2, K, P):
        EssentialMatrixInitializer.get_pose(q1, q2, K, P)

    @staticmethod
    def get_pose(q1, q2, K, P):
        """
        Predicts the transformation between two images using the essential matrix

        :param q1: The good keypoints matches position in i-1 th image
        :param q2: The good keypoints matches position in i th image
        :param K: The camera intrinsics
        :param P: The projection matrix
        :return: The transformation matrix - shape (4,3)
        """
        essential, mask = cv2.findEssentialMat(q1, q2, K)
        R, t = EssentialMatrixInitializer.decompose_essential_mat(essential, q1, q2, K, P)
        return rt_matrices_2_transform_matrix(R, t)

    @staticmethod
    def decompose_essential_mat(E, q1, q2, K, P1):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in ith image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """

        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = rt_matrices_2_transform_matrix(R1, t.flatten())
        T2 = rt_matrices_2_transform_matrix(R2, t.flatten())
        T3 = rt_matrices_2_transform_matrix(R1, -t.flatten())
        T4 = rt_matrices_2_transform_matrix(R2, -t.flatten())
        transformations = [T1, T2, T3, T4]

        # Homogenize K
        K = np.concatenate((K, np.zeros((3, 1))), axis=1)

        # List of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        np.set_printoptions(suppress=True)

        # print ("\nTransform 1\n" +  str(T1))
        # print ("\nTransform 2\n" +  str(T2))
        # print ("\nTransform 3\n" +  str(T3))
        # print ("\nTransform 4\n" +  str(T4))

        positives = []
        for P, T in zip(projections, transformations):
            hom_Q1 = cv2.triangulatePoints(P1, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
            positives.append(total_sum + relative_scale)

        # Decompose the Essential matrix using built-in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z

        max_ = np.argmax(positives)
        # print("positives: ", positives)

        if max_ == 2:
            return R1, -t.flatten()
        elif max_ == 3:
            return R2, -t.flatten()
        elif max_ == 0:
            return R1, t.flatten()
        elif max_ == 1:
            return R2, t.flatten()
