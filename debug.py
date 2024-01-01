import cv2
import matplotlib.pyplot as plt
from vslam_data import *


def debug_keypoint_matches(frame1: Frame, frame2: Frame, matches, ):
    """
    Plots the matches between two frames
    :param frame1:
    :param frame2:
    :param matches:
    :return:
    """
    keypoints1 = [
        cv2.KeyPoint(kp.x, kp.y, kp.size, kp.angle, class_id=kp.class_id, octave=kp.octave, response=kp.response) for kp
        in frame1.keypoints]
    keypoints2 = [
        cv2.KeyPoint(kp.x, kp.y, kp.size, kp.angle, class_id=kp.class_id, octave=kp.octave, response=kp.response) for kp
        in frame2.keypoints]
    d_matches = [cv2.DMatch(_queryIdx=q, _trainIdx=m, _distance=0.0, _imgIdx=0) for q, m in enumerate(matches) if
                 m != -1]
    # draw matches
    img_matches = cv2.drawMatches(frame1.image, keypoints1, frame2.image, keypoints2, d_matches, None,
                                  matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_matches)
    plt.show()
    return img_matches
