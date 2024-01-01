import math
import numpy as np
from vslam_data import Frame


class OrbInitializer:
    def __init__(self, sigma, iterations):
        # First Frame is reference frame
        # self.mK = reference_frame.mK.clone()
        # self.mvKeys1 = reference_frame.mvKeysUn
        self.mSigma = sigma
        self.mSigma2 = sigma * sigma
        self.mMaxIterations = iterations

    def initialize(self, first_frame: Frame, current_frame: Frame, initial_matches):
        """ Initialize the map given the first two frames and the initial matches
        1. Create mvMatches containing the matches between the reference and the current frame
        2. Generate sets of 8 points for each RANSAC iteration
        3. compute homography and fundamental matrix with scores
        4. Try to reconstruct from homography or fundamental depending on the score ratio (0.40-0.45)
        :param first_frame:
        :param current_frame:
        :param initial_matches:
        :return:
        """
        # Fill structures with current keypoints and matches with reference frame
        # Reference/First Frame: 1, Current Frame: 2
        # mvKeys1 = first_frame.keypoints
        # mvKeys2 = current_frame.keypoints

        indices_of_matched_points_pairs = []  # [(pair)] * len(self.mvKeys2)
        for index_of_point, index_of_matching_point in enumerate(initial_matches):
            if index_of_matching_point > -1:
                indices_of_matched_points_pairs.append((index_of_point, index_of_matching_point))
        number_of_matches = len(indices_of_matched_points_pairs)
        indices_of_matched_points_pairs = np.array(indices_of_matched_points_pairs)

        # Generate sets of 8 points for each RANSAC iteration
        number_of_points_for_ransac = 8
        if number_of_matches < number_of_points_for_ransac:
            print("Not enough matches to initialize")  # todo find case for its reverse not enough changes in frames
            return None, None, None, None, False
        ransac_sets = np.zeros((self.mMaxIterations, number_of_points_for_ransac), dtype=np.int32)
        generator = np.random.default_rng()
        for i in range(self.mMaxIterations):
            ransac_sets[i] = generator.choice(number_of_matches, size=number_of_points_for_ransac, replace=False)

        # find underlying homography or rigid transformation with matching points
        # find homography/rigid transform for all sets but return which satisfy the homography most for the matches
        # i.e. given homography/rigid transform most matched pair are inliers and produces highest score
        # Launch threads to compute in parallel a fundamental matrix and a homography
        match_pairs_satisfying_h_21, score_h, h_21 = self.find_homography(first_frame.keypoints,
                                                                          current_frame.keypoints,
                                                                          ransac_sets, indices_of_matched_points_pairs)
        match_pairs_satisfying_f_21, score_f, f_21 = self.find_fundamental(first_frame.keypoints,
                                                                           current_frame.keypoints,
                                                                           ransac_sets, indices_of_matched_points_pairs)

        # threadH = threading.Thread(target=self.find_homography, )  # args=(vbMatchesInliersH, SH, H)
        # threadF = threading.Thread(target=self.find_fundamental, )  # args=(vbMatchesInliersF, SF, F)
        # Wait until both threads have finished
        # threadH.join()
        # threadF.join()

        # Compute ratio of scores
        ratio_of_h = score_h / (score_h + score_f)

        # Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
        if ratio_of_h > 0.40:
            # find rotation and translation from homography matrix of frame2 wrt frame1 and find triangulated mappoints
            return self.reconstruct_h(match_pairs_satisfying_h_21, h_21, first_frame.keypoints, current_frame.keypoints,
                                      indices_of_matched_points_pairs, first_frame.intrinsics_matrix, 1.0, 50)
        else:  # if(pF_HF>0.6):
            # find rotation and translation from fundamental matrix of frame2 wrt frame1 and find triangulated mappoints
            # R21, t21, vP3D, vbTriangulated, is_initialized
            return self.reconstruct_f(match_pairs_satisfying_f_21, f_21, first_frame.keypoints, current_frame.keypoints,
                                      indices_of_matched_points_pairs, first_frame.intrinsics_matrix, 1.0, 50)

    def find_homography(self, keypoints1, keypoints2, ransac_sets, indices_of_matched_points):
        """
        Compute homography between two frames
        :param keypoints1:
        :param keypoints2:
        :param ransac_sets:
        :param indices_of_matched_points:
        :return: inliers, score and homography
        inliers - inliers of the RANSAC set with the highest score
        1. Normalize coordinates
        2. compute homography of a RANSAC set point wrt it's matched point
        3. check the homography matrix and calculate score of homography and inliers
        4. return inliers, score and homography of the RANSAC set with the highest score
        """

        # Normalize coordinates
        normalized_keypoints1, transform_keypoints1 = self.normalize(keypoints1)  # todo check vPn1
        normalized_keypoints2, transform_keypoints2 = self.normalize(keypoints2)
        transform_keypoints2_inv = np.linalg.inv(transform_keypoints2)

        # Best Results variables
        h_21 = np.zeros((3, 3))
        score = 0.0
        match_pairs_satisfying_h_21 = None

        # Iteration variables
        normalized_keypoint1_tmp = np.zeros((8, 2))
        normalized_keypoint2_tmp = np.zeros((8, 2))

        # todo check ransac algorithm
        # Perform all RANSAC iterations and save the solution with the highest score
        for it in range(self.mMaxIterations):
            # Select a minimum set
            for j in range(8):
                idx = ransac_sets[it, j]
                # todo check if it is correct. Todo explain and optimize
                normalized_keypoint1_tmp[j, :] = normalized_keypoints1[indices_of_matched_points[idx, 0], :]
                normalized_keypoint2_tmp[j, :] = normalized_keypoints2[indices_of_matched_points[idx, 1], :]

            h_temp = self.compute_h21(normalized_keypoint1_tmp, normalized_keypoint2_tmp)
            h_21_tmp = transform_keypoints2_inv @ h_temp @ transform_keypoints1  # T2inv @ Hn @ T1   todo check
            h_12_tmp = np.linalg.inv(h_21_tmp)

            current_score, matched_pair_satisfying_h21i = self.check_homography(h_21_tmp, h_12_tmp, keypoints1,
                                                                                keypoints2, indices_of_matched_points)

            if current_score > score:
                h_21 = h_21_tmp.copy()
                match_pairs_satisfying_h_21 = matched_pair_satisfying_h21i.copy()
                score = current_score

        return match_pairs_satisfying_h_21, score, h_21

    def find_fundamental(self, keypoints1, keypoints2, ransac_sets, indices_of_matched_points):
        """
        Compute fundamental matrix between two frames
        :param keypoints1:
        :param keypoints2:
        :param ransac_sets:
        :param indices_of_matched_points:
        :return:
        1. Normalize coordinates
        2. Compute fundamental matrix of a RANSAC set point wrt it's matched point
        3. Check the fundamental matrix and calculate score of fundamental matrix and inliers
        4. Return inliers, score and fundamental matrix of the RANSAC set with the highest score
        """
        # Number of putative matches
        number_of_putative_matches = len(indices_of_matched_points)

        # Normalize coordinates
        normalized_keypoints1, transform_keypoints1 = self.normalize(keypoints1)
        normalized_keypoints2, transform_keypoints2 = self.normalize(keypoints2)
        transform_keypoints2_transpose = transform_keypoints2.T  # Todo confirm by equation

        # Best Results variables
        f_21 = np.zeros((3, 3))
        score = 0.0
        match_pairs_satisfying_h_21 = np.array([False] * number_of_putative_matches, dtype=np.bool_)

        # Iteration variables
        normalized_keypoint1_tmp = np.zeros((8, 2))
        normalized_keypoint2_tmp = np.zeros((8, 2))

        # Perform all RANSAC iterations and save the solution with the highest score
        for it in range(self.mMaxIterations):
            # Select a minimum set
            for j in range(8):
                idx = ransac_sets[it][j]
                normalized_keypoint1_tmp[j] = normalized_keypoints1[indices_of_matched_points[idx, 0]]
                normalized_keypoint2_tmp[j] = normalized_keypoints2[indices_of_matched_points[idx, 1]]

            fundamental_matrix = self.compute_f21(normalized_keypoint1_tmp, normalized_keypoint2_tmp)
            f_21_tmp = transform_keypoints2_transpose @ fundamental_matrix @ transform_keypoints1
            current_score, matched_pair_satisfying_h21i = self.check_fundamental(f_21_tmp, keypoints1, keypoints2,
                                                                                 indices_of_matched_points)
            if current_score > score:
                f_21 = f_21_tmp
                match_pairs_satisfying_h_21 = matched_pair_satisfying_h21i
                score = current_score
        return match_pairs_satisfying_h_21, score, f_21

    # todo return messages instead of None when reconstruction fails
    def reconstruct_h(self, pairs_satisfying_h_21, h_21, keypoints1, keypoints2, indices_of_matched_points,
                      camera_intrinsics, min_parallax, min_triangulated_points):
        """

        :param pairs_satisfying_h_21: pairs which satisfy h_21 and can be used to filter pair which are not inliers
        :param h_21: homography matrix
        :param keypoints1: keypoints in first frame
        :param keypoints2: keypoints in current frame
        :param indices_of_matched_points: pairs matched by matcher
        :param camera_intrinsics: camera intrinsics
        :param min_parallax:
        :param min_triangulated_points:
        :return:
        1. Compute 8 motion hypotheses using the method of Faugeras et al.
        2. Triangulate 3d map points with the help of transformation matrices (R,t) and matched key points for 8 cases
        3. Return Camera pose Transform (R,t), and triangulated 3d map points locations, inliers and is_initialized
        """

        # We recover 8 motion hypotheses using the method of Faugeras et al.
        # Motion and structure from motion in a piecewise planar environment.
        # International Journal of Pattern Recognition and Artificial Intelligence, 1988
        rotation_matrices, translation_vectors, vn = (
            self.compute_transformation_from_h_faugeras_method(h_21, camera_intrinsics))

        if rotation_matrices is None or translation_vectors is None:
            return None, None, None, None, False

        best_good, second_best_good = 0, 0
        best_solution_idx, best_parallax = -1, -1
        best_p3d, best_triangulated = [], []
        # Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points
        # seen with low parallax) We reconstruct all hypotheses and check in terms of triangulated points and parallax
        # we have calculated 8 rotation - translation or 8 transformation matrix for pose of frame2 wrt frame1
        # triangulate 3d map points with the help of transformation matrix(R,t) and matched key points for all 8 cases
        for i in range(len(rotation_matrices)):  # 8 motion hypothesis
            # Triangulate
            p3d_tmp, bool_triangulated_tmp, parallax_i, n_good = self.check_rt(rotation_matrices[i],
                                                                               translation_vectors[i], keypoints1,
                                                                               keypoints2, indices_of_matched_points,
                                                                               pairs_satisfying_h_21, camera_intrinsics,
                                                                               4.0 * self.mSigma2)

            if n_good > best_good:
                second_best_good = best_good
                best_good = n_good
                best_solution_idx = i
                best_parallax = parallax_i
                best_p3d = p3d_tmp
                best_triangulated = bool_triangulated_tmp
            elif n_good > second_best_good:
                second_best_good = n_good

        number_of_pairs_satisfying_h_21 = [1 for i in range(len(pairs_satisfying_h_21)) if pairs_satisfying_h_21[i]]
        number_of_pairs_satisfying_h_21 = sum(number_of_pairs_satisfying_h_21)

        # second_best_good < 0.75 * best_good means second-best solution is not close to best solution i.e. by
        # triangulating at least 75% more than the second best
        print(f'{second_best_good < 0.75 * best_good=}')
        # best_parallax > min_parallax means the parallax of best rotation-translation pair is more than min_parallax
        print(f'{best_parallax > min_parallax=}')
        # best_good > min_triangulated_points means the number of triangulated points by best rotation-translation pair
        # is more than min_triangulated_points
        print(f'{best_good > min_triangulated_points=}')
        # best_good > 0.9 * number_of_pairs_satisfying_h_21 means more than 90% of matched pairs have been triangulated
        print(f'{best_good > 0.9 * number_of_pairs_satisfying_h_21=}')

        if (second_best_good < 0.75 * best_good and best_parallax > min_parallax and best_good > min_triangulated_points
                and best_good > 0.9 * number_of_pairs_satisfying_h_21):
            return rotation_matrices[best_solution_idx], translation_vectors[
                best_solution_idx], best_p3d, best_triangulated, True
        else:
            return None, None, None, None, False
        # return rcw, tcw, mvIniP3D, vbTriangulated, is_initialized

    def reconstruct_f(self, pairs_satisfying_f_21, f_21, keypoints1, keypoints2, indices_of_matched_points,
                      camera_intrinsics, min_parallax, min_triangulated):
        """
        :param pairs_satisfying_f_21:
        :param f_21:
        :param keypoints1:
        :param keypoints2:
        :param indices_of_matched_points:
        :param camera_intrinsics:
        :param min_parallax:
        :param min_triangulated:
        :return:
        1. Compute Essential Matrix from Fundamental Matrix
        2. Decompose Essential Matrix to get 4 possible transforms with R,t
        3. triangulate for 4 possible transforms and check which has most triangulated points and parallax
        4. return Camera pose Transform (R,t), and triangulated 3d map points locations, inliers and is_initialized
        """

        # Compute Essential Matrix from Fundamental Matrix
        essential_matrix_21 = camera_intrinsics.T @ f_21 @ camera_intrinsics
        rotation_matrix1, rotation_matrix2, translation_vector = self.decompose_e(essential_matrix_21)

        translation_vector1 = translation_vector
        translation_vector2 = -translation_vector

        # Reconstruct with the 4 hypothesis and check
        p3d1, bool_triangulated1, parallax1, n_good1 = self.check_rt(rotation_matrix1, translation_vector1, keypoints1,
                                                                     keypoints2, indices_of_matched_points,
                                                                     pairs_satisfying_f_21, camera_intrinsics,
                                                                     4.0 * self.mSigma2)
        p3d2, bool_triangulated2, parallax2, n_good2 = self.check_rt(rotation_matrix2, translation_vector1, keypoints1,
                                                                     keypoints2, indices_of_matched_points,
                                                                     pairs_satisfying_f_21, camera_intrinsics,
                                                                     4.0 * self.mSigma2)
        p3d3, bool_triangulated3, parallax3, n_good3 = self.check_rt(rotation_matrix1, translation_vector2, keypoints1,
                                                                     keypoints2, indices_of_matched_points,
                                                                     pairs_satisfying_f_21, camera_intrinsics,
                                                                     4.0 * self.mSigma2)
        p3d4, bool_triangulated4, parallax4, n_good4 = self.check_rt(rotation_matrix2, translation_vector2, keypoints1,
                                                                     keypoints2, indices_of_matched_points,
                                                                     pairs_satisfying_f_21, camera_intrinsics,
                                                                     4.0 * self.mSigma2)

        max_good = max(n_good1, n_good2, n_good3, n_good4)

        number_of_pairs_satisfying_f_21 = [1 for i in range(len(pairs_satisfying_f_21)) if pairs_satisfying_f_21[i]]
        number_of_pairs_satisfying_f_21 = sum(number_of_pairs_satisfying_f_21)
        min_good = max(int(0.9 * number_of_pairs_satisfying_f_21), min_triangulated)

        number_of_similar_rt_producing_results = 0
        if n_good1 > 0.7 * max_good:
            number_of_similar_rt_producing_results += 1
        if n_good2 > 0.7 * max_good:
            number_of_similar_rt_producing_results += 1
        if n_good3 > 0.7 * max_good:
            number_of_similar_rt_producing_results += 1
        if n_good4 > 0.7 * max_good:
            number_of_similar_rt_producing_results += 1

        # If there is not a clear winner or not enough triangulated points reject initialization
        print(f'{max_good < min_good=}')
        print(f'{number_of_similar_rt_producing_results > 1=}')
        if max_good < min_good or number_of_similar_rt_producing_results > 1:
            return None, None, None, None, False

        # If best reconstruction has enough parallax initialize
        if max_good == n_good1:
            if parallax1 > min_parallax:
                return rotation_matrix1, translation_vector1, p3d1, bool_triangulated1, True
        elif max_good == n_good2:
            if parallax2 > min_parallax:
                return rotation_matrix2, translation_vector1, p3d2, bool_triangulated2, True
        elif max_good == n_good3:
            if parallax3 > min_parallax:
                return rotation_matrix1, translation_vector2, p3d3, bool_triangulated3, True
        elif max_good == n_good4:
            if parallax4 > min_parallax:
                return rotation_matrix2, translation_vector2, p3d4, bool_triangulated4, True

        return None, None, None, None, False

    @staticmethod
    def compute_h21(normalized_keypoints1, normalized_keypoints2):
        """
        https://subscription.packtpub.com/book/data/9781789537147/1/ch01lvl1sec05/applying-perspective-transformation-and-homography
        :param normalized_keypoints1:
        :param normalized_keypoints2:
        :return:
        """
        number_of_pairs_in_ransac_set = len(normalized_keypoints1)
        a = np.zeros((2 * number_of_pairs_in_ransac_set, 9), dtype=np.float32)
        for i in range(number_of_pairs_in_ransac_set):
            u1, v1 = normalized_keypoints1[i]
            u2, v2 = normalized_keypoints2[i]

            a[2 * i, 3] = -u1
            a[2 * i, 4] = -v1
            a[2 * i, 5] = -1
            a[2 * i, 6] = v2 * u1
            a[2 * i, 7] = v2 * v1
            a[2 * i, 8] = v2

            a[2 * i + 1, 0] = u1
            a[2 * i + 1, 1] = v1
            a[2 * i + 1, 2] = 1
            a[2 * i + 1, 6] = -u2 * u1
            a[2 * i + 1, 7] = -u2 * v1
            a[2 * i + 1, 8] = -u2

        u, w, vt = np.linalg.svd(a, full_matrices=True)
        # vt.row(8).reshape(0, 3); # 0 channel(2d) 3 rows (3x3=9 left 3 columns) todo compile and check
        return vt[8].reshape(3, 3)

    @staticmethod
    def compute_f21(normalized_keypoints1, normalized_keypoints2):
        """

        :param normalized_keypoints1:
        :param normalized_keypoints2:
        :return:
        """
        number_of_pairs_in_ransac_set = len(normalized_keypoints1)
        a = np.zeros((number_of_pairs_in_ransac_set, 9), dtype=np.float32)
        for i in range(number_of_pairs_in_ransac_set):
            u1, v1 = normalized_keypoints1[i]
            u2, v2 = normalized_keypoints2[i]

            a[i, 0] = u2 * u1
            a[i, 1] = u2 * v1
            a[i, 2] = u2
            a[i, 3] = v2 * u1
            a[i, 4] = v2 * v1
            a[i, 5] = v2
            a[i, 6] = u1
            a[i, 7] = v1
            a[i, 8] = 1
        u, w, vt = np.linalg.svd(a, full_matrices=True)
        f_pre = vt[8].reshape((3, 3))  # vt.row(8).reshape(0, 3);
        u, w, vt = np.linalg.svd(f_pre, full_matrices=True)
        w[2] = 0  # w.at<float>(2)=0;  todo check correctness
        return u @ np.diag(w) @ vt  # todo check

    def check_homography(self, h21, h12, keypoints1, keypoints2, matched_pair_indices):
        """
        :param h21:
        :param h12:
        :param keypoints1:
        :param keypoints2:
        :param matched_pair_indices:
        :return: inliers, score
        inliers  is true for the match pair that have a reprojection error less than 5.991
        and image1 has points also presentH21i in image2 satisfy homography with h12 and h21 matrices
        given this homography which matched pair are inliers within certain threshold
        matched_pair_satisfying_h21
        """

        h11 = h21[0, 0]
        h_12 = h21[0, 1]
        h13 = h21[0, 2]
        h_21 = h21[1, 0]
        h22 = h21[1, 1]
        h23 = h21[1, 2]
        h31 = h21[2, 0]
        h32 = h21[2, 1]
        h33 = h21[2, 2]

        h11inv = h12[0, 0]
        h12inv = h12[0, 1]
        h13inv = h12[0, 2]
        h21inv = h12[1, 0]
        h22inv = h12[1, 1]
        h23inv = h12[1, 2]
        h31inv = h12[2, 0]
        h32inv = h12[2, 1]
        h33inv = h12[2, 2]

        number_of_matches = len(matched_pair_indices)
        matches_inliers = np.array([False] * number_of_matches, dtype=np.bool_)
        score = 0

        th = 5.991
        inv_sigma_square = 1.0 / self.mSigma2

        for i, match_pair in enumerate(matched_pair_indices):
            inlier = True
            kp1 = keypoints1[match_pair[0]]
            kp2 = keypoints2[match_pair[1]]

            u1 = kp1.x
            v1 = kp1.y
            u2 = kp2.x
            v2 = kp2.y
            # Reprojection error in first image
            #  x2in1 = h12 * x2
            w2in1inv = 1.0 / (h31inv * u2 + h32inv * v2 + h33inv)
            u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv
            v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv

            square_dist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1)
            chi_square1 = square_dist1 * inv_sigma_square

            if chi_square1 > th:
                inlier = False
            else:
                score += th - chi_square1

            # Reprojection error in second image
            # x1in2 = H21 * x1
            w1in2inv = 1.0 / (h31 * u1 + h32 * v1 + h33)
            u1in2 = (h11 * u1 + h_12 * v1 + h13) * w1in2inv
            v1in2 = (h_21 * u1 + h22 * v1 + h23) * w1in2inv

            square_dist2 = (u2 - u1in2) * (u2 - u1in2) + (v2 - v1in2) * (v2 - v1in2)
            chi_square2 = square_dist2 * inv_sigma_square

            if chi_square2 > th:
                inlier = False
            else:
                score += th - chi_square2

            if inlier:
                matches_inliers[i] = True
            else:
                matches_inliers[i] = False

        return score, matches_inliers

    def check_fundamental(self, f_21, keypoints1, keypoints2, matched_pair_indices):
        """

        :param f_21:
        :param keypoints1:
        :param keypoints2:
        :param matched_pair_indices:
        :return:
        """

        f11 = f_21[0][0]
        f12 = f_21[0][1]
        f13 = f_21[0][2]
        f21 = f_21[1][0]
        f22 = f_21[1][1]
        f23 = f_21[1][2]
        f31 = f_21[2][0]
        f32 = f_21[2][1]
        f33 = f_21[2][2]

        number_of_matches = len(matched_pair_indices)
        matches_inliers = np.array([False] * number_of_matches, dtype=np.bool_)
        score = 0

        th = 3.841
        th_score = 5.991
        inv_sigma_square = 1.0 / self.mSigma2

        for i in range(number_of_matches):
            inlier = True

            kp1 = keypoints1[matched_pair_indices[i][0]]
            kp2 = keypoints2[matched_pair_indices[i][1]]

            u1 = kp1.x
            v1 = kp1.y
            u2 = kp2.x
            v2 = kp2.y

            # Reprojection error in second image
            # l2=F21x1=(a2,b2,c2)
            a2 = f11 * u1 + f12 * v1 + f13
            b2 = f21 * u1 + f22 * v1 + f23
            c2 = f31 * u1 + f32 * v1 + f33

            num2 = a2 * u2 + b2 * v2 + c2
            square_dist1 = num2 ** 2 / (a2 ** 2 + b2 ** 2)
            chi_square1 = square_dist1 * inv_sigma_square

            if chi_square1 > th:
                inlier = False
            else:
                score += th_score - chi_square1

            # Reprojection error in second image
            # l1 =x2tF21=(a1,b1,c1)
            a1 = f11 * u2 + f21 * v2 + f31
            b1 = f12 * u2 + f22 * v2 + f32
            c1 = f13 * u2 + f23 * v2 + f33

            num1 = a1 * u1 + b1 * v1 + c1
            square_dist2 = num1 ** 2 / (a1 ** 2 + b1 ** 2)
            chi_square2 = square_dist2 * inv_sigma_square

            if chi_square2 > th:
                inlier = False
            else:
                score += th_score - chi_square2

            if inlier:  # todo check for any bin False i.e any chiSquare2 or chiSquare1 < th this will not run
                matches_inliers[i] = True
            else:
                matches_inliers[i] = False

        return score, matches_inliers

    @staticmethod
    def decompose_e(essential_matrix):
        u, w, vt = np.linalg.svd(essential_matrix, full_matrices=True)  # todo check full_matrices=True
        translation_vector = u[:, 2]
        translation_vector = translation_vector / np.linalg.norm(translation_vector)

        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        rotation_matrix1 = u @ W @ vt
        if np.linalg.det(rotation_matrix1) < 0:
            rotation_matrix1 = -rotation_matrix1
        rotation_matrix2 = u @ W.T @ vt
        if np.linalg.det(rotation_matrix2) < 0:
            rotation_matrix2 = -rotation_matrix2
        return rotation_matrix1, rotation_matrix2, translation_vector.reshape(3, 1)

    # todo check correctness, rename and optimize loops
    def check_rt(self, rotation_matrix, translation_vector, keypoints1, keypoints2, matched_pair_indices,
                 pair_satisfying_hf_that_produced_rt, camera_intrinsics, th2):
        """
        Check the Rt matrix by triangulating the points and checking parallax

        :param rotation_matrix:
        :param translation_vector:
        :param keypoints1:
        :param keypoints2:
        :param matched_pair_indices:
        :param pair_satisfying_hf_that_produced_rt:
        :param camera_intrinsics:
        :param th2:
        :return:
        """
        # Calibration parameters
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]

        good_p3d = [False] * len(keypoints1)
        p3d_locations = [[]] * len(keypoints1)
        cos_parallax_lst = []  # populated to length of vKeys1

        # Camera 1 Projection Matrix K[I|0]
        projection_matrix1 = np.zeros((3, 4))
        projection_matrix1[0:3, 0:3] = camera_intrinsics  # [0:3, 0:3] todo check

        O1 = np.zeros((3, 1))

        # Camera 2 Projection Matrix K[R|t]
        projection_matrix2 = np.zeros((3, 4))
        projection_matrix2[0:3, 0:3] = rotation_matrix
        projection_matrix2[0:3, 3:4] = translation_vector  # 3:4 is done to assign col 3 a 3x1 matrix
        projection_matrix2 = camera_intrinsics @ projection_matrix2

        O2 = -rotation_matrix.T @ translation_vector

        nGood = 0
        for i in range(len(matched_pair_indices)):
            # if pair has not satisfied h or not inlier then continued here
            if not pair_satisfying_hf_that_produced_rt[i]:
                continue

            kp1 = keypoints1[matched_pair_indices[i, 0]]  # todo check from above
            kp2 = keypoints2[matched_pair_indices[i, 1]]
            p3d_location1 = self.triangulate(kp1, kp2, projection_matrix1, projection_matrix2)

            if not np.all(np.isfinite(p3d_location1)):  # if all are not finite then continue
                good_p3d[matched_pair_indices[i, 0]] = False  # todo redundant if initialized with False
                continue

            # Check parallax
            normal1 = p3d_location1 - O1
            dist1 = np.linalg.norm(normal1)

            normal2 = p3d_location1 - O2
            dist2 = np.linalg.norm(normal2)

            cos_parallax = normal1.T @ normal2 / (dist1 * dist2)  # todo its normal1.dot(normal2) in cpp run and confirm
            cos_parallax = cos_parallax.item()

            # Check depth in front of first camera (only if enough parallax, as
            # "infinite" points can easily go to negative depth)
            if p3d_location1[2] <= 0 and cos_parallax < 0.99998:
                continue

            # Check depth in front of second camera (only if enough parallax, as
            # "infinite" points can easily go to negative depth)
            p3d_location2 = rotation_matrix @ p3d_location1 + translation_vector

            if p3d_location2[2] <= 0 and cos_parallax < 0.99998:
                continue

            # Check reprojection error in first image
            inv_z1 = 1.0 / p3d_location1[2]
            im1x = fx * p3d_location1[0] * inv_z1 + cx
            im1y = fy * p3d_location1[1] * inv_z1 + cy

            square_error1 = (im1x - kp1.x) ** 2 + (im1y - kp1.y) ** 2

            if square_error1 > th2:
                continue

            # Check reprojection error in second image
            inv_z2 = 1.0 / p3d_location2[2]
            im2x = fx * p3d_location2[0] * inv_z2 + cx
            im2y = fy * p3d_location2[1] * inv_z2 + cy

            square_error2 = (im2x - kp2.x) ** 2 + (im2y - kp2.y) ** 2

            if square_error2 > th2:
                continue

            cos_parallax_lst.append(cos_parallax)
            p3d_locations[matched_pair_indices[i, 0]] = p3d_location1  # todo correct assignment with append
            nGood += 1

            if cos_parallax < 0.99998:
                good_p3d[matched_pair_indices[i, 0]] = True

        if nGood > 0:
            cos_parallax_lst.sort()
            idx = min(50, len(cos_parallax_lst) - 1)
            parallax = math.acos(cos_parallax_lst[idx]) * 180 / np.pi  # todo check what it does and what it should do
        else:
            parallax = 0
        # vP3D, vbGood, parallax, nGood
        return p3d_locations, good_p3d, parallax, nGood

    @staticmethod
    def normalize(keypoints):
        kps_len = len(keypoints)
        mean_x = sum([kp.x for kp in keypoints]) / kps_len
        mean_y = sum([kp.y for kp in keypoints]) / kps_len

        normalized_points = np.zeros((kps_len, 2))

        for i in range(kps_len):
            normalized_points[i, 0] = keypoints[i].x - mean_x
            normalized_points[i, 1] = keypoints[i].y - mean_y

        mean_dev_x = np.mean(np.abs(normalized_points[:, 0]))
        mean_dev_y = np.mean(np.abs(normalized_points[:, 1]))

        s_x = 1.0 / mean_dev_x
        s_y = 1.0 / mean_dev_y

        normalized_points[:, 0] = normalized_points[:, 0] * s_x
        normalized_points[:, 1] = normalized_points[:, 1] * s_y

        transform = np.eye(3, 3, dtype=np.float32)
        transform[0, 0] = s_x
        transform[1, 1] = s_y
        transform[0, 2] = -mean_x * s_x
        transform[1, 2] = -mean_y * s_y
        return normalized_points, transform

    @staticmethod
    def triangulate(kp1, kp2, camera1_projection, camera2_projection):
        a = np.zeros((4, 4), dtype=np.float32)

        a[0, :] = kp1.x * camera1_projection[2, :] - camera1_projection[0, :]
        a[1, :] = kp1.y * camera1_projection[2, :] - camera1_projection[1, :]
        a[2, :] = kp2.x * camera2_projection[2, :] - camera2_projection[0, :]
        a[3, :] = kp2.y * camera2_projection[2, :] - camera2_projection[1, :]

        u, w, vt = np.linalg.svd(a, full_matrices=True)
        x3d = vt[3, :].T
        x3d = x3d[0:3] / x3d[3]
        return x3d.reshape(3, 1)

    @staticmethod
    def compute_transformation_from_h_faugeras_method(h_21, camera_intrinsics):
        inv_k = np.linalg.inv(camera_intrinsics)
        a = inv_k @ h_21 @ camera_intrinsics

        U, w, Vt = np.linalg.svd(a, full_matrices=True)
        V = Vt.T

        s = np.linalg.det(U) * np.linalg.det(Vt)

        d1 = w[0]
        d2 = w[1]
        d3 = w[2]

        rotation_matrices, translation_vectors, vn = [], [], []
        if d1 / d2 < 1.00001 or d2 / d3 < 1.00001:
            return None, None, None

        # n'=[x1 0 x3] 4 possibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
        aux1 = math.sqrt((d1 * d1 - d2 * d2) / (d1 * d1 - d3 * d3))
        aux3 = math.sqrt((d2 * d2 - d3 * d3) / (d1 * d1 - d3 * d3))
        x1 = [aux1, aux1, -aux1, -aux1]
        x3 = [aux3, -aux3, aux3, -aux3]

        # case d'=d2
        aux_s_theta = math.sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2)

        c_theta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2)
        s_theta = [aux_s_theta, -aux_s_theta, -aux_s_theta, aux_s_theta]

        for i in range(4):
            Rp = np.eye(3, 3, dtype=np.float32)
            Rp[0, 0] = c_theta
            Rp[0, 2] = -s_theta[i]
            Rp[2, 0] = s_theta[i]
            Rp[2, 2] = c_theta

            rotation_matrix = s * U @ Rp @ Vt
            rotation_matrices.append(rotation_matrix)

            tp = np.zeros((3, 1))
            tp[0] = x1[i]
            tp[1] = 0
            tp[2] = -x3[i]
            tp *= d1 - d3

            translation_vector = U @ tp
            translation_vectors.append(translation_vector / np.linalg.norm(translation_vector))

            n_p = np.zeros((3, 1))
            n_p[0] = x1[i]
            n_p[1] = 0
            n_p[2] = x3[i]

            n = V @ n_p
            if n[2] < 0:
                n = -n
            vn.append(n)

        # case d'=-d2
        aux_s_phi = math.sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2)
        c_phi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2)
        s_phi = [aux_s_phi, -aux_s_phi, -aux_s_phi, aux_s_phi]

        for i in range(4):
            Rp = np.eye(3, dtype=np.float32)
            Rp[0, 0] = c_phi
            Rp[0, 2] = s_phi[i]
            Rp[1, 1] = -1
            Rp[2, 0] = s_phi[i]
            Rp[2, 2] = -c_phi

            rotation_matrix = s * U @ Rp @ Vt
            rotation_matrices.append(rotation_matrix)

            tp = np.zeros((3, 1), dtype=np.float32)
            tp[0] = x1[i]
            tp[1] = 0
            tp[2] = x3[i]
            tp *= d1 + d3

            translation_vector = U @ tp
            translation_vectors.append(translation_vector / np.linalg.norm(translation_vector))

            n_p = np.zeros((3, 1), dtype=np.float32)
            n_p[0] = x1[i]
            n_p[1] = 0
            n_p[2] = x3[i]

            n = V @ n_p
            if n[2] < 0:
                n = -n
            vn.append(n)
        return rotation_matrices, translation_vectors, vn
