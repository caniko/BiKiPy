from functools import cached_property
from typing import Optional

import pandas as pd
from pydantic_numpy import NDArrayBool

from bikipy.feature.qualia.axioms.proximity import ComputeProximity
from bikipy.feature.qualia.physical_object.heuristic.abc import (
    AbstractQualiaHeuristic,
    ProximityMixin,
)
from bikipy.feature.tolerance.plural import plural_node_tolerance_model


class BodyProximityHeuristic(AbstractQualiaHeuristic, ProximityMixin):
    center_eye_label: str | None = "center_eye"
    torso_label: str | None
    base_tail_label: str | None = "base_tail"

    manual_center_eye: Optional[ComputeProximity]
    manual_torso: Optional[ComputeProximity]
    manual_base_tail: Optional[ComputeProximity]

    heuristic_alias = "BodyProximity"

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("manual_center_eye", "manual_torso", "manual_base_tail"))
        return upstream

    @cached_property
    def center_eye_proximity(self) -> ComputeProximity | None:
        if self.manual_center_eye is not None:
            return self.manual_center_eye

        if not self.center_eye_label:
            return None

        return ComputeProximity(
            perimeter=self.perimeter,
            perimeter_border_normal_pixels=self.maximum_distance_pixels,
            should_be_inside_perimeter_border=self.reader[self.center_eye_label],
            label="Center eye",
            manual_video=self.video,
        )

    @cached_property
    def torso_proximity(self) -> ComputeProximity | None:
        if self.manual_torso is not None:
            return self.manual_torso

        if not self.torso_label:
            return None

        return ComputeProximity(
            perimeter=self.perimeter,
            perimeter_border_normal_pixels=self.maximum_distance_pixels,
            should_be_inside_perimeter_border=self.reader[self.torso_label],
            label="Torso",
            manual_video=self.video,
        )

    @cached_property
    def base_tail_proximity(self) -> ComputeProximity | None:
        if self.manual_base_tail is not None:
            return self.manual_base_tail

        if not self.base_tail_label:
            return None

        return ComputeProximity(
            perimeter=self.perimeter,
            perimeter_border_normal_pixels=self.maximum_distance_pixels,
            should_be_inside_perimeter_border=self.reader[self.base_tail_label],
            label="Base tail",
            manual_video=self.video,
        )

    @property
    def result(self) -> NDArrayBool:
        return (
            plural_node_tolerance_model(
                self.center_eye_proximity.result,
                self.torso_proximity.result,
                self.base_tail_proximity.result,
                fps=self.video.fps,
            )
            if self.filter_in_sequence
            else self.center_eye_proximity.result | self.torso_proximity.result | self.base_tail_proximity.result
        )

    @property
    def summary_series(self) -> pd.Series:
        return pd.Series(
            {
                "CenterEyeProximity": self.center_eye_proximity.result,
                "TorsoProximity": self.torso_proximity.result,
                "BaseTailProximity": self.base_tail_proximity.result,
                self.heuristic_alias: self.result,
            }
        )

    def plot(self) -> None:
        fig, axes = self.video.subplots(ncols=4, nrows=1)

        if self.center_eye_proximity:
            self.center_eye_proximity.plot(axes[0], self.video)

        if self.torso_proximity:
            self.torso_proximity.plot(axes[1], self.video)

        if self.base_tail_proximity:
            self.base_tail_proximity.plot(axes[2], self.video)

        self.plot_result(axes[3])
