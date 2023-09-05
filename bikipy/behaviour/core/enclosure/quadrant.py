from functools import cached_property

import numpy as np
from pydantic import computed_field
from pydantic_numpy.typing import NpNDArrayBool, NpNDArrayFp64

from bikipy.core.base import BikipyModel
from bikipy.core.video import VideoMetadata
from bikipy.feature.motion import merge_motion_island_data
from bikipy.math.confinement.polygon import parallel_point_inside_polygon
from bikipy.math.discrete import (
    TruthIslandMetadata,
    tolerance_modeled_boolean_index_truth_sequence_start_end_length,
)
from bikipy.math.geometry import clockwise_sort_points


class Quadrant(BikipyModel):
    vertices_in_meters: NpNDArrayFp64
    kinematic_coordinates: NpNDArrayFp64
    fps: float
    quadrant_index: int

    @computed_field  # type: ignore[misc]
    @cached_property
    def _motion_island_confinement_boolean_index(self) -> tuple[TruthIslandMetadata, NpNDArrayBool]:
        return tolerance_modeled_boolean_index_truth_sequence_start_end_length(
            parallel_point_inside_polygon(self.kinematic_coordinates, clockwise_sort_points(self.vertices_in_meters)),
            self.fps,
        )

    def plot_vertices(self, video: VideoMetadata) -> NpNDArrayFp64:
        return video.prepare_coordinates_for_plotting(self.vertices_in_meters)

    @computed_field  # type: ignore[misc]
    @property
    def confinement_boolean_index(self) -> NpNDArrayBool:
        return self._motion_island_confinement_boolean_index[1]

    @computed_field  # type: ignore[misc]
    @cached_property
    def seconds_present(self) -> float:
        return np.sum(self.confinement_boolean_index) / self.fps

    @computed_field  # type: ignore[misc]
    @property
    def motion(self) -> dict[str, float]:
        return merge_motion_island_data(
            self._motion_island_confinement_boolean_index[0], self.kinematic_coordinates, self.fps
        )
