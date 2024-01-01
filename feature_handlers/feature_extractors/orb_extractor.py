import cv2
from typing import List
from vslam_data import KeyPoint


class ORBFeatureExtractor:
    def __init__(self, number_of_features=1000, scale_factor=1.2, number_of_levels=8, edge_threshold=31, first_level=0,
                 wta_k=2, patch_size=31, fast_threshold=20):
        self.orb = cv2.ORB_create(nfeatures=number_of_features, scaleFactor=scale_factor, nlevels=number_of_levels,
                                  edgeThreshold=edge_threshold, firstLevel=first_level, WTA_K=wta_k,
                                  patchSize=patch_size, fastThreshold=fast_threshold)

    def extract(self, image) -> List[KeyPoint]:
        keypoints = []
        for kp, desc in zip(*self.orb.detectAndCompute(image, None)):
            key_pt = KeyPoint(1, x=kp.pt[0], y=kp.pt[1], angle=kp.angle, class_id=kp.class_id, octave=kp.octave,
                              response=kp.response, size=kp.size, descriptor=desc)
            keypoints.append(key_pt)
        return keypoints
