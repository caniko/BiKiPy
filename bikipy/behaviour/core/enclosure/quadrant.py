from functools import cached_property

import numpy as np
from pydantic_numpy import NDArrayFp64, NDArrayBool

from bikipy.core.base_class import BaseBikipy
from bikipy.feature.motion import get_combined_features_from_merged_motion_island_data
from bikipy.utils.math.geometry import clockwise_sort_points
from bikipy.utils.math.inside.polygon import parallel_point_inside_polygon


class Quadrant(BaseBikipy):
    vertices_in_meters: NDArrayFp64
    kinematic_coordinates: NDArrayFp64
    fps: float
    quadrant_index: int

    @cached_property
    def confinement_boolean_index(self) -> NDArrayBool:
        return parallel_point_inside_polygon(self.kinematic_coordinates, clockwise_sort_points(self.vertices_in_meters))

    @cached_property
    def seconds_present(self) -> float:
        return np.sum(self.confinement_boolean_index) / self.fps

    @cached_property
    def motion(self) -> dict[str, float]:
        return get_combined_features_from_merged_motion_island_data(
            self.confinement_boolean_index,
            self.kinematic_coordinates,
            self.fps,
        )
