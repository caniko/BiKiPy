from functools import cached_property
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from pydantic import computed_field
from pydantic_numpy.typing import Np1DArrayBool, NpNDArrayFp64

from bikipy.feature.qualia.axioms.ilos import ComputeInLineOfSight
from bikipy.feature.qualia.axioms.proximity import ComputeProximity
from bikipy.feature.qualia.physical_object.heuristic.mixin import (
    ProximityMixin,
    RayMixin,
)
from bikipy.feature.qualia.physical_object.heuristic.solo.abc import (
    AbstractSoloHeuristic,
)
from bikipy.math.discrete import start_all_true_end_main_false
from bikipy.perimeter.base import BasePerimeter


class OlfactionHeuristic(AbstractSoloHeuristic, ProximityMixin, RayMixin):
    maximum_distance_meters: float = 0.05
    maximum_degrees: float = 60.0

    nose_direction_naive: bool = True

    nose_label: str | None = "nose"
    center_ear_label: str = "center_ear"

    manual_nose: Optional[ComputeProximity] = None
    manual_nose_olfaction_rays: Optional[ComputeInLineOfSight] = None

    heuristic_alias = "Olfaction"

    @classmethod
    @property
    def fields_to_exclude_from_single_schema(cls) -> set[str]:
        result = super().fields_to_exclude_from_single_schema
        result.update(("manual_nose", "manual_torso", "manual_tail_base"))
        return result

    @computed_field  # type: ignore[misc]
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

    @computed_field  # type: ignore[misc]
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

    @computed_field  # type: ignore[misc]
    @property
    def solo_result(self) -> Np1DArrayBool:
        if self.nose_direction_naive:
            return start_all_true_end_main_false(self.nose_proximity.result, self.snout_towards_object_rays.result)

        return self.nose_proximity.result & self.snout_towards_object_rays.result

    @computed_field  # type: ignore[misc]
    @property
    def label_to_proximity_boolean(self) -> dict[str, Np1DArrayBool]:
        return {self.nose_label: self.nose_proximity.result}

    @computed_field  # type: ignore[misc]
    @property
    def label_to_ray_vector_direction_points(self) -> dict[str, NpNDArrayFp64]:
        return {self.nose_label: self.reader[self.nose_label] - self.reader[self.center_ear_label]}

    @computed_field  # type: ignore[misc]
    @property
    def perimeter_to_boolean_index(self) -> dict[BasePerimeter, Np1DArrayBool]:
        return self.nose_proximity.video_gen_merge_perimeter_to_boolean_index(
            self.snout_towards_object_rays, both_or_false=True
        )

    @computed_field  # type: ignore[misc]
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

    def plot(self) -> plt.Figure:
        fig, axes = self.video.subplots(nrows=3, title=self.heuristic_alias)

        self.nose_proximity.plot(axes[0], self.video)
        self.snout_towards_object_rays.plot(axes[1], self.video)
        self.plot_result(axes[2], self.nose_label)

        return fig
