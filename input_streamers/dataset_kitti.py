import glob
import numpy as np
import os
from PIL import Image


__all__ = 'KittiDataset',


class KittiDataset:
    def __init__(self, root_dir, monocular=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.monocular = monocular

        self.K, self.P = KittiDataset.load_calib(os.path.join(root_dir, 'calib.txt'))
        self.gt_poses = KittiDataset.load_poses(os.path.join(root_dir, 'poses.txt'))
        self.image_left_path_list = sorted(glob.glob(os.path.join(root_dir, 'image_l/*.png')))
        assert len(self.image_left_path_list) == len(
            self.gt_poses), f"Number of images , {len(self.image_left_path_list)} != {len(self.gt_poses)}"
        if not monocular:
            self.image_rpath_list = sorted(glob.glob(os.path.join(root_dir, 'image_r/*.png')))
            assert len(self.image_left_path_list) == len(self.image_rpath_list)

    @staticmethod
    def load_calib(calib_filepath):
        with open(calib_filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            camera_intrinsics_k = P[0:3, 0:3]
            return camera_intrinsics_k, P

    @staticmethod
    def load_poses(poses_filepath):
        poses = []
        with open(poses_filepath, 'r') as f:
            for line in f.readlines():
                pose = np.fromstring(line, dtype=np.float64, sep=' ')
                pose = pose.reshape(3, 4)
                pose = np.vstack((pose, [0, 0, 0, 1]))
                poses.append(pose)
            return poses

    def __len__(self):
        return len(self.gt_poses)

    def __getitem__(self, idx):
        image_l = np.array(Image.open(self.image_left_path_list[idx]))
        if self.monocular:
            return self.K, self.P, self.gt_poses[idx], image_l
        else:
            image_r = np.array(Image.open(self.image_rpath_list[idx]))
            return self.K, self.P, self.gt_poses[idx], image_l, image_r
