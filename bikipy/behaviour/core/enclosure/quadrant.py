from functools import cached_property

import numpy as np
from pydantic_numpy import NDArrayFp64

from bikipy.core.base import BikipyModel
from bikipy.core.video import VideoMetadata
from bikipy.feature.motion import merge_motion_island_data
from bikipy.utils.math.discrete import (
    TruthIslandMetadata,
    tolerance_modeled_boolean_index_truth_sequence_start_end_length,
)
from bikipy.utils.math.geometry import clockwise_sort_points
from bikipy.utils.math.confinement.polygon import parallel_point_inside_polygon


class Quadrant(BikipyModel):
    vertices_in_meters: NDArrayFp64
    kinematic_coordinates: NDArrayFp64
    fps: float
    quadrant_index: int

    @cached_property
    def _motion_island_confinement_boolean_index(self) -> tuple[TruthIslandMetadata, np.ndarray[bool, bool]]:
        return tolerance_modeled_boolean_index_truth_sequence_start_end_length(
            parallel_point_inside_polygon(self.kinematic_coordinates, clockwise_sort_points(self.vertices_in_meters)),
            self.fps,
        )

    def plot_vertices(self, video: VideoMetadata) -> np.ndarray[float, np.dtype[np.float64]]:
        return video.prepare_coordinates_for_plotting(self.vertices_in_meters)

    @property
    def confinement_boolean_index(self) -> np.ndarray[bool, bool]:
        return self._motion_island_confinement_boolean_index[1]

    @cached_property
    def seconds_present(self) -> float:
        return np.sum(self.confinement_boolean_index) / self.fps

    @property
    def motion(self) -> dict[str, float]:
        return merge_motion_island_data(
            self._motion_island_confinement_boolean_index[0], self.kinematic_coordinates, self.fps
        )
