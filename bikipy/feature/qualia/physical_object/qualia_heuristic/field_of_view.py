from functools import cached_property
from typing import Optional

import pandas as pd
from pydantic_numpy import NDArrayBool

from bikipy.feature.qualia.axioms.proximity import ComputeProximity
from bikipy.feature.qualia.axioms.ray import ComputeRay
from bikipy.feature.qualia.physical_object.qualia_heuristic.abc import (
    AbstractQualiaProfile,
    ProximityMixin,
    RayMixin,
)
from bikipy.feature.tolerance.plural import plural_node_tolerance_model
from bikipy.feature.tolerance.single import single_node_tolerance_model


class FOVCenterToEyesRayCastingProfile(AbstractQualiaProfile, ProximityMixin, RayMixin):
    center_eye_label: str = "center_ear"
    left_eye_label: str = "left_ear"
    right_eye_label: str = "right_ear"

    manual_left_eye_proximity: Optional[ComputeProximity]
    manual_leftward_observation: Optional[ComputeRay]
    manual_right_proximity: Optional[ComputeProximity]
    manual_rightward_observation: Optional[ComputeRay]

    profile_alias = "ObjectInProximalFOV"

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(
            (
                "manual_left_eye_proximity",
                "manual_leftward_observation",
                "manual_right_proximity",
                "manual_rightward_observation",
            )
        )
        return upstream

    @cached_property
    def left_proximity(self) -> ComputeProximity:
        return (
            self.manual_left_eye_proximity
            if self.manual_left_eye_proximity
            else ComputeProximity(
                perimeter=self.perimeter,
                perimeter_border_normal_pixels=self.maximum_distance_pixels,
                inside_perimeter_border=self.reader[self.left_eye_label],
                manual_video=self.video,
            )
        )

    @cached_property
    def leftward_observation(self) -> ComputeRay:
        return (
            self.manual_leftward_observation
            if self.manual_leftward_observation
            else ComputeRay(
                perimeter=self.perimeter,
                ray_start_point=self.reader[self.left_eye_label],
                ray_travel_direction_point=self.reader[self.center_eye_label],
                max_radians=self.maximum_radians,
                manual_video=self.video,
            )
        )

    @cached_property
    def right_proximity(self) -> ComputeProximity:
        return (
            self.manual_right_proximity
            if self.manual_right_proximity
            else ComputeProximity(
                perimeter=self.perimeter,
                perimeter_border_normal_pixels=self.maximum_distance_pixels,
                inside_perimeter_border=self.reader[self.right_eye_label],
                manual_video=self.video,
            )
        )

    @cached_property
    def rightward_observation(self) -> ComputeRay:
        return (
            self.manual_rightward_observation
            if self.manual_rightward_observation
            else ComputeRay(
                perimeter=self.perimeter,
                ray_start_point=self.reader[self.right_eye_label],
                ray_travel_direction_point=self.reader[self.center_eye_label],
                max_radians=self.maximum_radians,
                manual_video=self.video,
            )
        )

    @cached_property
    def left_result(self) -> NDArrayBool:
        result = self.left_proximity.result & self.leftward_observation.result
        return result if self.filter_in_sequence else single_node_tolerance_model(result, self.fps)

    @cached_property
    def right_result(self) -> NDArrayBool:
        result = self.right_proximity.result & self.rightward_observation.result
        return result if self.filter_in_sequence else single_node_tolerance_model(result, self.fps)

    @cached_property
    def result(self) -> NDArrayBool:
        return (
            plural_node_tolerance_model(self.left_result, self.right_result, fps=self.fps)
            if self.filter_in_sequence
            else self.left_result | self.right_result
        )

    @property
    def summary_series(self) -> pd.Series:
        return pd.Series(
            {
                "LeftProximity": self.left_proximity.result_seconds,
                "LeftwardFOV": self.leftward_observation.result_seconds,
                "RightProximity": self.right_proximity.result_seconds,
                "RightwardFOV": self.rightward_observation.result_seconds,
                self.profile_alias: self.result,
            }
        )

    def plot(self) -> None:
        fig, axes = self.video.subplots(ncols=3, nrows=3)

        # Left
        self.left_proximity.plot(axes[0][0], self.video)
        self.leftward_observation.plot(axes[0][1], self.video)

        axes[0][2].set_title("LeftwardProximalFOV")
        self.reader.plot_boolean_index(self.left_result, axes[0][2])

        # Right
        self.right_proximity.plot(axes[1][0], self.video)
        self.rightward_observation.plot(axes[1][1], self.video)

        axes[1][2].set_title("RightwardProximalFOV")
        self.reader.plot_boolean_index(self.right_result, axes[1][2])

        self.plot_result(axes[2][1])
