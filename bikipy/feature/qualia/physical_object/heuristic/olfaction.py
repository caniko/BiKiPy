from functools import cached_property
from typing import Optional

import pandas as pd
from pydantic_numpy import NDArrayBool

from bikipy.feature.qualia.axioms.proximity import ComputeProximity
from bikipy.feature.qualia.physical_object.heuristic.abc import AbstractQualiaHeuristic, ProximityMixin


class OlfactionHeuristic(AbstractQualiaHeuristic, ProximityMixin):
    nose_label: str | None = "nose"
    torso_label: str | None = "torso"

    manual_nose: Optional[ComputeProximity]

    heuristic_alias = "BodyProximity"

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("manual_nose", "manual_torso", "manual_tail_base"))
        return upstream

    @cached_property
    def nose_proximity(self) -> ComputeProximity | None:
        if self.manual_nose is not None:
            return self.manual_nose

        if not self.nose_label:
            return None

        return ComputeProximity(
            perimeter=self.perimeter,
            perimeter_border_normal_pixels=self.maximum_distance_pixels,
            should_be_inside_perimeter_border=self.reader[self.nose_label],
            should_be_outside_perimeter_border=self.reader[self.torso_label],
            label="Center eye",
            manual_video=self.video,
        )

    @cached_property
    def result(self) -> NDArrayBool:
        return self.nose_proximity.result

    @property
    def summary_series(self) -> pd.Series:
        label = self.perimeter.label.capitalize()
        return pd.Series(self.nose_proximity.result_seconds, index=[f"ObservingSecNoseProximity{label}"])

    def plot(self) -> None:
        fig, axes = self.video.subplots()

        ax_idx = 0

        if self.nose_proximity:
            axes[ax_idx].set_title("Center eye")
            self.nose_proximity.plot(axes[ax_idx], self.video)

        self.plot_result(axes[ax_idx], self.nose_label)
