import cv2
import sys
from vslam_data import Frame


class ORBMatcher:
    def __init__(self, window_size=100, mb_check_orientation=True, mf_n_ratio=0.9, th_low=50):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        flann_index_lsh = 6
        index_params = dict(algorithm=flann_index_lsh, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=4500)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.TH_LOW = th_low
        self.window_size, self.mb_check_orientation, self.mf_n_ratio = window_size, mb_check_orientation, mf_n_ratio

    def match(self, prev_keypoints, prev_descriptors, keypoints, descriptors):
        # return self.bf.match(descriptors1, descriptors2)
        matches = self.flann.knnMatch(prev_descriptors, descriptors, k=2)
        # Apply ratio test to filter good matches
        good_matches = []
        for pair in matches:
            try:
                if pair[0].distance < 0.9 * pair[1].distance:
                    good_matches.append(pair[0])
            except ValueError:
                pass
        prev_matched_points = [prev_keypoints[m.queryIdx] for m in good_matches]
        matched_points = [keypoints[m.trainIdx] for m in good_matches]
        return prev_matched_points, matched_points, good_matches

    def match_init(self, frame1: Frame, frame2: Frame):
        """
        SearchForInitialization equivalent
        :param frame1:
        :param frame2:
        :return:
        """
        HISTO_LENGTH = 30
        matches = 0
        rot_hist = [[] for _ in range(HISTO_LENGTH)]
        factor = 1.0 / HISTO_LENGTH

        vb_prev_matched = frame2.keypoints
        keypoints_f1 = frame1.keypoints
        keypoints_f2 = frame2.keypoints

        vn_matches12 = [-1] * len(keypoints_f1)
        v_matched_distance = [sys.maxsize] * len(keypoints_f2)
        vn_matches21 = [-1] * len(keypoints_f2)

        for i1 in range(len(keypoints_f1)):
            kp1 = keypoints_f1[i1]
            level1 = kp1.octave
            if level1 > 0:
                continue

            #  Get features near point in frame2
            v_indices2 = frame2.get_features_in_area(vb_prev_matched[i1].x, vb_prev_matched[i1].y, self.window_size,
                                                     level1, level1)

            if len(v_indices2) == 0:
                continue

            d1 = kp1.descriptor

            best_dist = sys.maxsize
            best_dist2 = sys.maxsize
            best_idx2 = -1

            for vit in v_indices2:
                i2 = vit

                d2 = keypoints_f2[i2].descriptor

                dist = cv2.norm(d1, d2, cv2.NORM_HAMMING)

                if v_matched_distance[i2] <= dist:
                    continue

                if dist < best_dist:
                    best_dist2 = best_dist
                    best_dist = dist
                    best_idx2 = i2
                elif dist < best_dist2:
                    best_dist2 = dist

            if best_dist <= self.TH_LOW:
                if best_dist < best_dist2 * self.mf_n_ratio:
                    if vn_matches21[best_idx2] >= 0:
                        vn_matches12[vn_matches21[best_idx2]] = -1
                        matches -= 1
                    vn_matches12[i1] = best_idx2
                    vn_matches21[best_idx2] = i1
                    v_matched_distance[best_idx2] = best_dist
                    matches += 1

                    if self.mb_check_orientation:
                        rot = keypoints_f1[i1].angle - keypoints_f2[best_idx2].angle
                        if rot < 0.0:
                            rot += 360.0
                        bin_ = round(rot * factor)
                        if bin_ == HISTO_LENGTH:
                            bin_ = 0
                        assert 0 <= bin_ < HISTO_LENGTH
                        rot_hist[bin_].append(i1)

        if self.mb_check_orientation:
            ind1, ind2, ind3 = ORBMatcher.compute_three_maxima(rot_hist, HISTO_LENGTH)

            for i in range(HISTO_LENGTH):
                if i == ind1 or i == ind2 or i == ind3:
                    continue
                for j in range(len(rot_hist[i])):
                    idx1 = rot_hist[i][j]
                    if vn_matches12[idx1] >= 0:
                        vn_matches12[idx1] = -1
                        matches -= 1

        # Update prev matched
        for i1 in range(len(vn_matches12)):
            if vn_matches12[i1] >= 0:
                vb_prev_matched[i1] = keypoints_f2[vn_matches12[i1]]

        return matches, vn_matches12

    @staticmethod
    def compute_three_maxima(histogram, histo_length):
        max1 = 0
        max2 = 0
        max3 = 0
        ind1, ind2, ind3 = -1, -1, -1
        for i in range(histo_length):
            s = len(histogram[i])
            if s > max1:
                max3 = max2
                max2 = max1
                max1 = s
                ind3 = ind2
                ind2 = ind1
                ind1 = i
            elif s > max2:
                max3 = max2
                max2 = s
                ind3 = ind2
                ind2 = i
            elif s > max3:
                max3 = s
                ind3 = i

        if max2 < 0.1 * max1:
            ind2 = -1
            ind3 = -1
        elif max3 < 0.1 * max1:
            ind3 = -1
        return ind1, ind2, ind3
