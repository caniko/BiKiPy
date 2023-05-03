from functools import cached_property
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pydantic_numpy import NDArrayBool

from bikipy.feature.attention.proximity import ComputeProximity
from bikipy.feature.attention.ray import ComputeRay
from bikipy.feature.physical_object.qualia.component.abc import AbstractQualiaComponent
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.utils.math.cached import cached_deg2rad, meters2pixels


class FOVCenterToEyesRayCasting(AbstractQualiaComponent):
    center_eye_label: str = "center_ear"
    left_eye_label: str = "left_ear"
    right_eye_label: str = "right_ear"

    manual_left_eye_proximity: Optional[ComputeProximity]
    manual_leftward_observation: Optional[ComputeRay]
    manual_right_proximity: Optional[ComputeProximity]
    manual_rightward_observation: Optional[ComputeRay]

    maximum_gaze_distance_meters: float = 0.05
    gaze_maximum_degrees: float = 45.0

    @property
    def gaze_length_pixels(self) -> float:
        return meters2pixels(self.maximum_gaze_distance_meters, self.video.pixels_per_meter)

    @property
    def gaze_maximum_radians(self) -> float:
        return cached_deg2rad(self.gaze_maximum_degrees)

    @cached_property
    def left_proximity(self) -> ComputeProximity:
        return (
            self.manual_left_eye_proximity
            if self.manual_left_eye_proximity
            else ComputeProximity(
                perimeter=self.perimeter,
                perimeter_border_normal_pixels=self.gaze_length_pixels,
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
                max_radians=self.gaze_maximum_radians,
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
                perimeter_border_normal_pixels=self.gaze_length_pixels,
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
                max_radians=self.gaze_maximum_radians,
                manual_video=self.video,
            )
        )

    @cached_property
    def result(self) -> NDArrayBool:
        left = self.left_proximity.result & self.leftward_observation.result
        right = self.right_proximity.result & self.rightward_observation.result
        if self.filter_in_sequence:
            return np.logical_or(
                single_node_tolerance_model(left, self.video.fps), single_node_tolerance_model(right, self.video.fps)
            )
        return left | right

    @property
    def summary_series(self) -> pd.Series:
        return pd.Series(
            {
                "LeftProximity": self.left_proximity.result_seconds,
                "LeftwardFOV": self.leftward_observation.result_seconds,
                "RightProximity": self.right_proximity.result_seconds,
                "RightwardFOV": self.rightward_observation.result_seconds,
                "ObjectInProximalFOV": self.result,
            }
        )

    def plot(self, axes: Optional[Sequence[Axes]] = None) -> None:
        fig, axes = self.video.subplots(ncols=5)

        self.left_proximity.plot(axes[0], self.video)
        self.leftward_observation.plot(axes[1], self.video)
        self.right_proximity.plot(axes[2], self.video)
        self.rightward_observation.plot(axes[3], self.video)
        self.reader.plot_boolean_index(self.result, axes[4])
