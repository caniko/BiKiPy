from functools import cached_property
from typing import Optional

import pandas as pd
from pydantic import computed_field
from pydantic_numpy import NpNDArrayFp64
from pydantic_numpy.typing import NpNDArrayBool

from bikipy.core.compute import video_gen_merge_perimeter_to_boolean_index_from_dict
from bikipy.feature.qualia.axioms.ilos import ComputeInLineOfSight
from bikipy.feature.qualia.axioms.proximity import ComputeProximity
from bikipy.feature.qualia.physical_object.heuristic.mixin import (
    ProximityMixin,
    RayMixin,
)
from bikipy.feature.qualia.physical_object.heuristic.solo.abc import (
    AbstractSoloHeuristic,
)
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import Perimeter


class WhiskerInteractionHeuristic(AbstractSoloHeuristic, ProximityMixin, RayMixin):
    maximum_distance_meters = 0.035

    center_ear_label: str = "center_ear"
    left_ear_label: str = "left_ear"
    right_ear_label: str = "right_ear"

    manual_left_ear_proximity: Optional[ComputeProximity] = None
    manual_leftward_observation: Optional[ComputeInLineOfSight] = None
    manual_right_proximity: Optional[ComputeProximity] = None
    manual_rightward_observation: Optional[ComputeInLineOfSight] = None

    heuristic_alias = "WhiskerInteraction"

    @computed_field(return_type=set[str])  # type: ignore[misc]
    @classmethod
    @property
    def fields_to_exclude_from_single_schema(cls) -> set[str]:
        result = super().fields_to_exclude_from_single_schema
        result.update(
            (
                "manual_left_ear_proximity",
                "manual_leftward_observation",
                "manual_right_proximity",
                "manual_rightward_observation",
            )
        )
        return result

    @computed_field  # type: ignore[misc]
    @cached_property
    def left_proximity(self) -> ComputeProximity:
        return (
            self.manual_left_ear_proximity
            if self.manual_left_ear_proximity
            else ComputeProximity(
                perimeter=self.perimeter,
                maximum_distance=self.maximum_distance_pixels,
                inside_perimeter_border=self.reader[self.left_ear_label],
                label="Left",
                manual_video=self.video,
            )
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def leftward_observation(self) -> ComputeInLineOfSight:
        return (
            self.manual_leftward_observation
            if self.manual_leftward_observation
            else ComputeInLineOfSight(
                perimeter=self.perimeter,
                ray_start_point=self.reader[self.center_ear_label],
                ray_travel_direction_point=self.reader[self.left_ear_label],
                max_radians=self.maximum_radians,
                label="Left",
                manual_video=self.video,
            )
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def right_proximity(self) -> ComputeProximity:
        return (
            self.manual_right_proximity
            if self.manual_right_proximity
            else ComputeProximity(
                perimeter=self.perimeter,
                maximum_distance=self.maximum_distance_pixels,
                inside_perimeter_border=self.reader[self.right_ear_label],
                label="Right",
                manual_video=self.video,
            )
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def rightward_observation(self) -> ComputeInLineOfSight:
        return (
            self.manual_rightward_observation
            if self.manual_rightward_observation
            else ComputeInLineOfSight(
                perimeter=self.perimeter,
                ray_start_point=self.reader[self.center_ear_label],
                ray_travel_direction_point=self.reader[self.right_ear_label],
                max_radians=self.maximum_radians,
                label="Right",
                manual_video=self.video,
            )
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def left_result(self) -> NpNDArrayBool:
        result = self.left_proximity.result & self.leftward_observation.result
        return result if self.filter_in_sequence else single_node_tolerance_model(result, self.video.fps)

    @computed_field  # type: ignore[misc]
    @cached_property
    def right_result(self) -> NpNDArrayBool:
        result = self.right_proximity.result & self.rightward_observation.result
        return result if self.filter_in_sequence else single_node_tolerance_model(result, self.video.fps)

    @computed_field  # type: ignore[misc]
    @property
    def solo_result(self) -> NpNDArrayBool:
        return self.left_result | self.right_result

    @computed_field  # type: ignore[misc]
    @property
    def label_to_proximity_boolean(self) -> dict[str, NpNDArrayBool]:
        return {self.left_ear_label: self.left_proximity.result, self.right_ear_label: self.right_proximity.result}

    @computed_field  # type: ignore[misc]
    @property
    def label_to_ray_vector_direction_points(self) -> dict[str, NpNDArrayFp64]:
        return {
            self.left_ear_label: self.reader[self.left_ear_label] - self.reader[self.center_ear_label],
            self.right_ear_label: self.reader[self.right_ear_label] - self.reader[self.center_ear_label],
        }

    @computed_field  # type: ignore[misc]
    @property
    def perimeter_to_boolean_index(self) -> dict[Perimeter, NpNDArrayBool]:
        return video_gen_merge_perimeter_to_boolean_index_from_dict(
            self.left_proximity.video_gen_merge_perimeter_to_boolean_index(
                self.leftward_observation, both_or_false=True
            ),
            self.right_proximity.video_gen_merge_perimeter_to_boolean_index(
                self.rightward_observation, both_or_false=True
            ),
        )

    @computed_field  # type: ignore[misc]
    @property
    def summary_series(self) -> pd.Series:
        label = self.perimeter.label.capitalize()
        return pd.Series(
            (
                self.video.boolean_array_to_seconds(self.left_result),
                self.video.boolean_array_to_seconds(self.right_result),
                self.video.boolean_array_to_seconds(self.result),
            ),
            index=[f"{label}LeftwardProxFOV", f"{label}RightwardProxFOV", f"{label}{self.heuristic_alias}Result"],
        )

    def plot(self) -> None:
        fig, axes = self.video.subplots(ncols=3, nrows=3, exclude_imaging_from_rc_coord=((0, 2), (2, 2)))
        fig.suptitle(self.heuristic_alias)

        # Left
        self.left_proximity.plot(axes[0][0], self.video)
        self.leftward_observation.plot(axes[0][1], self.video)

        axes[0][2].set_title("LeftwardProximalFOV")
        self.reader.plot_boolean_index(self.left_result, axes[0][2], self.left_ear_label)

        # Right
        self.right_proximity.plot(axes[1][0], self.video)
        self.rightward_observation.plot(axes[1][1], self.video)

        axes[1][2].set_title("RightwardProximalFOV")
        self.reader.plot_boolean_index(self.right_result, axes[1][2], self.right_ear_label)

        self.plot_result(axes[2][1], self.center_ear_label)
