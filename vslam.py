import sys
import input_streamers
from debug import debug_keypoint_matches
from feature_handlers.feature_extractors.orb_extractor import ORBFeatureExtractor
from feature_handlers.featrure_matchers.orb_matcher import ORBMatcher
import initializers
import os
from tqdm.contrib import tenumerate
from vslam_data import *

DEBUG = os.environ.get('DEBUG', False)
if DEBUG:
    import cv2


class VSLAM:
    """
    Visual SLAM class main class that handles the whole pipeline
    """

    class Camera:
        Monocular = 1
        Stereo = 2
        Depth = 3

    def create_vslam(self, config):
        """Create VSLAM object based on yaml config"""
        pass

    def __init__(self, save_dir, data_stream, feature_extractor, feature_matcher, initializer):
        self.vslam_data = VSLAMData()

        self.data_stream = data_stream
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.initializer = initializer

        self.save_dir = save_dir
        self.camera_type = VSLAM.Camera.Monocular
        self.is_initialized = False
        self.MAX_FRAMES_FOR_INITIALIZATION = 100  # should be smaller than length of dataset
        self.frames_for_initialization = 0  # number of frames used for initialization

    def initialize_map_and_pose(self, first_frame: Frame, current_frame: Frame):
        # feature handler
        number_of_matches, vn_matches = self.feature_matcher.match_init(first_frame, current_frame)
        if DEBUG: debug_keypoint_matches(first_frame, current_frame, vn_matches)
        # initializer
        rcw, tcw, p3d_points, p3d_successfully_triangulated, is_initialized = (
            self.initializer.initialize(first_frame, current_frame, vn_matches))
        # current_frame.pose = rt_matrices_2_transform_matrix(rcw, tcw)
        # CreateInitialMapMonocular()
        # if mvIniMatches[i] >= 0 and not vbTriangulated[i]: # reduce non points triangulated in mvIniMatches
        #     mvIniMatches[i] = -1
        #     matches -= 1
        self.is_initialized = is_initialized
        self.frames_for_initialization += 1
        return self.is_initialized

    def localize(self):
        # feature handler
        # localizer
        pass

    def map(self):
        pass

    def optimize(self):
        pass

    def loop_closure(self):
        # feature handler
        # loop closure
        pass

    def evaluate(self):
        pass

    def visualize(self):
        pass

    def save(self):
        pass

    def run(self):
        if self.camera_type != VSLAM.Camera.Monocular:
            print("Only Monocular camera is supported. Exiting...")
            sys.exit(0)

        # input streamer
        prev_k, prev_p, prev_pose, *prev_images = self.data_stream[0]
        # extract first frame features
        prev_keypoints = self.feature_extractor.extract(prev_images[0])
        # prepare first Frame
        prev_frame = Frame(frame_id=0, image=prev_images[0], gt_pose=prev_pose, keypoints=prev_keypoints,
                           intrinsics_matrix=prev_k, projection_matrix=prev_p)

        # start data feed to SLAM
        for frame_number, (K, P, current_pose, *images) in tenumerate(self.data_stream, initial=1, start=1, unit='fra'):
            # feature extractor
            keypoints = self.feature_extractor.extract(images[0])
            current_frame = Frame(frame_id=frame_number, image=images[0], gt_pose=current_pose, keypoints=keypoints,
                                  intrinsics_matrix=K, projection_matrix=P)

            # feature handler can be used here in loop or inside respective methods for subtasks
            if not self.is_initialized:
                slam_initialized = self.initialize_map_and_pose(prev_frame, current_frame)
                if frame_number >= self.MAX_FRAMES_FOR_INITIALIZATION:
                    print("Unable to Initialize. Try increasing MAX_FRAMES_FOR_INITIALIZATION. Exiting...")
                    sys.exit(0)
                print(f"Initialized: {slam_initialized}")
            else:  # when initialized
                pass
                # self.localize(images)  # parallel
                # self.map(images)  # parallel
                # self.optimize()  # parallel
                # self.loop_closure()  # parallel

            # update previous frame
            if self.is_initialized:  # this is to keep first frame as previous frame if not initialized
                prev_frame = current_frame

        self.save()
        self.evaluate()
        self.visualize()


if __name__ == '__main__':
    # initialize components
    data_dir = os.environ.get('KITTI_PATH', False)
    dataset = input_streamers.KittiDataset(data_dir)
    extractor = ORBFeatureExtractor()
    matcher = ORBMatcher()
    # initializer = initializers.EssentialMatrixInitializer()
    initializer = initializers.OrbInitializer(1.0, 200)

    # instantiate vslam
    vslam = VSLAM('./save', dataset, extractor, matcher, initializer)
    vslam.run()
