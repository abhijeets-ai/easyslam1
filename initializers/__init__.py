from .essential_matrix_initializer import EssentialMatrixInitializer
from .h_f_initializer import HFInitializer
from .orb_initializer import OrbInitializer

__all__ = ['EssentialMatrixInitializer', 'HFInitializer', 'OrbInitializer']


class BaseInitializer:
    def __init__(self, save_dir, dataloader, feature, map_data):
        self.save_dir = save_dir
        self.dataloader = dataloader
        self.feature = feature
        self.map_data = map_data

    def initialize(self):
        raise NotImplementedError
