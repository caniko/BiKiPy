from functools import cached_property
from typing import Optional

import numpy as np
import pandas as pd

from bikipy.feature.qualia.axioms.ilos import ComputeInLineOfSight
from bikipy.feature.qualia.axioms.proximity import ComputeProximity
from bikipy.feature.qualia.physical_object.heuristic.mixin import ProximityMixin, RayMixin
from bikipy.feature.qualia.physical_object.heuristic.solo.abc import AbstractSoloHeuristic


class OlfactionHeuristic(AbstractSoloHeuristic, ProximityMixin, RayMixin):
    maximum_distance_meters = 0.05
    maximum_degrees = 45.0

    nose_label: str | None = "nose"
    center_ear_label: str = "center_ear"

    manual_nose: Optional[ComputeProximity]
    manual_nose_olfaction_rays: Optional[ComputeInLineOfSight]

    heuristic_alias = "Olfaction"

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        result = super().schemantic_fields_to_exclude_from_config_schema
        result.update(("manual_nose", "manual_torso", "manual_tail_base"))
        return result

    @cached_property
    def nose_proximity(self) -> ComputeProximity | None:
        if self.manual_nose is not None:
            return self.manual_nose

        if not self.nose_label:
            return None

        return ComputeProximity(
            perimeter=self.perimeter,
            maximum_distance=self.maximum_distance_meters,
            inside_perimeter_border=self.reader[self.nose_label],
            label="NoseProximity",
            manual_video=self.video,
        )

    @cached_property
    def snout_towards_object_rays(self) -> ComputeInLineOfSight:
        return (
            self.manual_nose_olfaction_rays
            if self.manual_nose_olfaction_rays
            else ComputeInLineOfSight(
                perimeter=self.perimeter,
                ray_start_point=self.reader[self.center_ear_label],
                ray_travel_direction_point=self.reader[self.nose_label],
                max_radians=self.maximum_radians,
                label="SnoutTowardsObject",
                manual_video=self.video,
            )
        )

    @property
    def solo_result(self) -> np.ndarray[bool, bool]:
        return self.nose_proximity.result & self.snout_towards_object_rays.result

    @property
    def summary_series(self) -> pd.Series:
        label = self.perimeter.label.capitalize()
        return pd.Series(
            [self.nose_proximity.result_seconds, self.snout_towards_object_rays.result_seconds],
            index=[f"ObservingSecNoseProximity{label}", ""],
        )

    @property
    def label_to_proximity_boolean(self) -> dict[str, np.ndarray[bool, bool]]:
        return {self.nose_label: self.nose_proximity.result}

    @property
    def label_to_ray_vector_direction_points(self) -> dict[str, np.ndarray[float, np.dtype[np.float64]]]:
        return {self.nose_label: self.reader[self.nose_label] - self.reader[self.center_ear_label]}

    def plot(self) -> None:
        fig, axes = self.video.subplots(nrows=3)
        fig.suptitle(self.heuristic_alias)

        self.nose_proximity.plot(axes[0], self.video)
        self.snout_towards_object_rays.plot(axes[1], self.video)
        self.plot_result(axes[2], self.nose_label)
