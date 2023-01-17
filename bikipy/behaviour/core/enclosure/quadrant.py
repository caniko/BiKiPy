from functools import cached_property

import numpy as np
from pydantic_numpy import NDArrayBool, NDArrayFp64

from bikipy.core.base_class import BikipyModel
from bikipy.core.video import VideoMetadata
from bikipy.feature.motion import get_combined_features_from_merged_motion_island_data
from bikipy.utils.math.geometry import clockwise_sort_points
from bikipy.utils.math.inside.polygon import parallel_point_inside_polygon


class Quadrant(BikipyModel):
    vertices_in_meters: NDArrayFp64
    kinematic_coordinates: NDArrayFp64
    fps: float
    quadrant_index: int

    def plot_vertices(self, video: VideoMetadata) -> NDArrayFp64:
        return video.prepare_coordinates_for_plotting(self.vertices_in_meters)

    @cached_property
    def confinement_boolean_index(self) -> NDArrayBool:
        return parallel_point_inside_polygon(
            self.reader.kinematic_coordinates, clockwise_sort_points(self.vertices_in_meters)
        )

    @cached_property
    def seconds_present(self) -> float:
        return np.sum(self.confinement_boolean_index) / self.fps

    @cached_property
    def motion(self) -> dict[str, float]:
        return get_combined_features_from_merged_motion_island_data(
            self.confinement_boolean_index,
            self.reader.kinematic_coordinates,
            self.fps,
        )
