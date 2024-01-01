from dataclasses import dataclass
from typing import Optional, Any
import cv2
from math import floor, ceil, fabs
import networkx as nx
import numpy as np

__all__ = ('MapPoint', 'KeyPoint', 'Frame', 'KeyFrame', 'EssentialGraph', 'CovisibilityGraph', 'SpanningTree',
           'VSLAMData')


@dataclass
class MapPoint:
    mappoint_id: int
    x: int
    y: int
    z: int
    descriptor: np.ndarray


# keypoint methods overlap, convert
@dataclass
class KeyPoint:
    keypoint_id: int
    x: float
    y: float
    angle: int
    class_id: int
    octave: int
    response: float
    size: float
    descriptor: np.ndarray


@dataclass
class EssentialGraph:
    graph = nx.Graph  # may not be needed as it is already in networkx


@dataclass
class CovisibilityGraph:
    graph = nx.Graph  # may not be needed as it is already in networkx


@dataclass
class SpanningTree:
    pass


@dataclass
class Frame:
    FRAME_GRID_ROWS = 48
    FRAME_GRID_COLS = 64

    frame_id: int
    image: Any
    gt_pose: np.ndarray  # with respect to world
    keypoints: list[KeyPoint]
    intrinsics_matrix: np.ndarray  # camera intrinsics
    projection_matrix: np.ndarray  # camera projection matrix

    # reference_keyframe: KeyFrame
    next_frame_id: Optional[int] = None
    features: Optional[np.ndarray] = None
    mappoints: Optional[list[MapPoint]] = None

    translation_matrix: Optional[np.ndarray] = None  # with respect to world
    rotation_matrix: Optional[np.ndarray] = None  # with respect to world
    pose: Optional[np.ndarray] = None  # with respect to world
    # camera features class including distance coefficient, calibration matrix
    distortion_coefficients: Optional[np.ndarray] = None
    # grid
    min_x: Optional[float] = None
    max_x: Optional[float] = None
    min_y: Optional[float] = None
    max_y: Optional[float] = None
    grid_element_height_inv: Optional[float] = None
    grid_element_width_inv: Optional[float] = None
    grid: Optional[list[list[list[int]]]] = None  # grid of keypoints

    def __post_init__(self):
        self.assign_features_to_grid()

    def position_in_grid(self, kp):
        pos_x = round((kp.x - self.min_x) * self.grid_element_width_inv)
        pos_y = round((kp.y - self.min_y) * self.grid_element_height_inv)

        # Keypoint's coordinates are undistorted, which could cause to go out of the image
        if pos_x < 0 or pos_x >= self.FRAME_GRID_COLS or pos_y < 0 or pos_y >= self.FRAME_GRID_ROWS:
            return None, None

        return int(pos_x), int(pos_y)

    def assign_features_to_grid(self):
        self.compute_image_bounds()
        n_reserve = int(0.5 * len(self.keypoints) / (self.FRAME_GRID_COLS * self.FRAME_GRID_ROWS))
        self.grid = [[[] for _ in range(self.FRAME_GRID_ROWS)] for _ in range(self.FRAME_GRID_COLS)]

        for i in range(len(self.keypoints)):
            kp = self.keypoints[i]

            grid_pos_x, grid_pos_y = self.position_in_grid(kp)
            if grid_pos_x is not None and grid_pos_y is not None:
                # if len(self.grid[grid_pos_x][grid_pos_y]) < n_reserve:
                self.grid[grid_pos_x][grid_pos_y].append(i)

    def is_in_frustum(self):
        pass

    def get_features_in_area(self, x, y, r, min_level, max_level):
        v_indices = []
        min_cell_x = max(0, int(floor((x - self.min_x - r) * self.grid_element_width_inv)))
        if min_cell_x >= self.FRAME_GRID_COLS:
            return v_indices

        max_cell_x = min(self.FRAME_GRID_COLS - 1, int(ceil((x - self.min_x + r) * self.grid_element_width_inv)))
        if max_cell_x < 0:
            return v_indices

        min_cell_y = max(0, int(floor((y - self.min_y - r) * self.grid_element_height_inv)))
        if min_cell_y >= self.FRAME_GRID_ROWS:
            return v_indices

        max_cell_y = min(self.FRAME_GRID_ROWS - 1, int(ceil((y - self.min_y + r) * self.grid_element_height_inv)))
        if max_cell_y < 0:
            return v_indices

        b_check_levels = (min_level > 0) or (max_level >= 0)

        for ix in range(min_cell_x, max_cell_x + 1):
            for iy in range(min_cell_y, max_cell_y + 1):
                v_cell = self.grid[ix][iy]
                if not v_cell:
                    continue

                for j in range(len(v_cell)):
                    kp_un = self.keypoints[v_cell[j]]
                    if b_check_levels:
                        if kp_un.octave < min_level:
                            continue
                        if max_level >= 0:
                            if kp_un.octave > max_level:
                                continue

                    dist_x = kp_un.x - x
                    dist_y = kp_un.y - y

                    if fabs(dist_x) < r and fabs(dist_y) < r:
                        v_indices.append(v_cell[j])

        return v_indices

    def un_distort_key_points(self):
        pass

    def compute_image_bounds(self):
        """
        Computes the image bounds for the given self
        :param self:
        :return: min_x, max_x, min_y, max_y
        """
        if self.distortion_coefficients and self.distortion_coefficients[0] != 0.0:
            bounds = np.zeros((4, 2), dtype=np.float32)
            bounds[1, 0] = self.image.shape[1]
            bounds[2, 1] = self.image.shape[0]
            bounds[3, 0] = self.image.shape[1]
            bounds[3, 1] = self.image.shape[0]

            # Undistorted corners
            bounds = bounds.reshape((2, 4)).T
            bounds = cv2.undistortPoints(bounds, self.intrinsics_matrix, self.distortion_coefficients,
                                         P=self.intrinsics_matrix)
            bounds = bounds.reshape(1, -1, 2)

            self.min_x = min(bounds[0, 0, 0], bounds[0, 2, 0])
            self.max_x = max(bounds[0, 1, 0], bounds[0, 3, 0])
            self.min_y = min(bounds[0, 0, 1], bounds[0, 1, 1])
            self.max_y = max(bounds[0, 2, 1], bounds[0, 3, 1])

        else:
            self.min_x = 0.0
            self.max_x = self.image.shape[1]
            self.min_y = 0.0
            self.max_y = self.image.shape[0]
        self.FRAME_GRID_ROWS = Frame.FRAME_GRID_ROWS
        self.FRAME_GRID_COLS = Frame.FRAME_GRID_COLS
        self.grid_element_height_inv = Frame.FRAME_GRID_ROWS / (self.max_y - self.min_y)
        self.grid_element_width_inv = Frame.FRAME_GRID_COLS / (self.max_x - self.min_x)
        return self.min_x, self.max_x, self.min_y, self.max_y

    def compute_stereo_matches(self):
        pass

    def compute_stereo_from_rgbd(self):
        pass

    def un_project_stereo(self):
        pass

    def extract_orb(self, image):
        pass

    def compute_bow(self, image):
        pass


@dataclass
class KeyFrame(Frame):
    keyframe_id: int = None
    next_keyframe_id: int = None
    # parent:KeyFrame
    # children: list[KeyFrame]


class VSLAMData:
    map_points: list[MapPoint]
    keyframes: list[KeyFrame]
    trajectory_poses: list[np.ndarray]
    transforms: list[np.ndarray]
    essential_graph: EssentialGraph
    covisibility_graph: CovisibilityGraph
    kf_spanning_tree: SpanningTree
