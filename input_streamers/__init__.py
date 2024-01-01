from .dataset_kitti import KittiDataset
from vslam_data import Frame  # adds dependency from different package TODO: try avoid this
__all__ = (KittiDataset,)


class BaseDataStreamer:
    def __init__(self,):
        pass

    def get_data(self) -> Frame:
        raise NotImplementedError
