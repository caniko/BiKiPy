from functools import cached_property
from typing import Optional

import pandas as pd
from pydantic import computed_field

from bikipy.feature.qualia.axioms.ilos import ComputeInLineOfSight
from bikipy.feature.qualia.axioms.proximity import ComputeProximity
from bikipy.feature.qualia.physical_object.heuristic.mixin import (
    ProximityMixin,
    RayMixin,
)
from bikipy.feature.qualia.physical_object.heuristic.solo.abc import (
    AbstractSoloHeuristic,
)
from bikipy.perimeter.base import Perimeter


class OlfactionHeuristic(AbstractSoloHeuristic, ProximityMixin, RayMixin):
    maximum_distance_meters = 0.05
    maximum_degrees = 45.0

    nose_label: str | None = "nose"
    center_ear_label: str = "center_ear"

    manual_nose: Optional[ComputeProximity]
    manual_nose_olfaction_rays: Optional[ComputeInLineOfSight]

    heuristic_alias = "Olfaction"

    @computed_field(return_type=set[str])
    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        result = super().schemantic_fields_to_exclude_from_config_schema
        result.update(("manual_nose", "manual_torso", "manual_tail_base"))
        return result

    @computed_field
    @cached_property
    def nose_proximity(self) -> ComputeProximity | None:
        if self.manual_nose is not None:
            return self.manual_nose

        if not self.nose_label:
            return None

        return ComputeProximity(
            perimeter=self.perimeter,
            maximum_distance=self.maximum_distance_pixels,
            inside_perimeter_border=self.reader[self.nose_label],
            label="NoseProximity",
            manual_video=self.video,
        )

    @computed_field
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

    @computed_field
    @property
    def solo_result(self) -> NpNDArrayBool:
        return self.nose_proximity.result & self.snout_towards_object_rays.result

    @computed_field
    @property
    def label_to_proximity_boolean(self) -> dict[str, NpNDArrayBool]:
        return {self.nose_label: self.nose_proximity.result}

    @computed_field
    @property
    def label_to_ray_vector_direction_points(self) -> dict[str, NpNDArrayFp64]:
        return {self.nose_label: self.reader[self.nose_label] - self.reader[self.center_ear_label]}

    @computed_field
    @property
    def perimeter_to_boolean_index(self) -> dict[Perimeter, NpNDArrayBool]:
        return self.nose_proximity.video_gen_merge_perimeter_to_boolean_index(
            self.snout_towards_object_rays, both_or_false=True
        )

    @computed_field
    @property
    def summary_series(self) -> pd.Series:
        label = self.perimeter.label.capitalize()
        return pd.Series(
            [
                self.nose_proximity.result_seconds,
                self.snout_towards_object_rays.result_seconds,
                self.video.boolean_array_to_seconds(self.result),
            ],
            index=[f"{label}NoseProximity", f"{label}NoseToObjectRay", f"{label}{self.heuristic_alias}Result"],
        )

    def plot(self) -> None:
        fig, axes = self.video.subplots(nrows=3)
        fig.suptitle(self.heuristic_alias)

        self.nose_proximity.plot(axes[0], self.video)
        self.snout_towards_object_rays.plot(axes[1], self.video)
        self.plot_result(axes[2], self.nose_label)
