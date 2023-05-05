from functools import cached_property
from typing import Optional

import pandas as pd
from pydantic_numpy import NDArrayBool

from bikipy.feature.qualia.axioms.proximity import ComputeProximity
from bikipy.feature.qualia.physical_object.qualia_heuristic.abc import (
    AbstractQualiaProfile,
    ProximityMixin,
)
from bikipy.feature.tolerance.plural import plural_node_tolerance_model


class BodyProximityProfile(AbstractQualiaProfile, ProximityMixin):
    center_eye_label: str = "center_ear"
    torso_label: str = "torso"
    base_tail_label: str = "base_tail"

    manual_center_eye: Optional[ComputeProximity]
    manual_torso: Optional[ComputeProximity]
    manual_tail_label: Optional[ComputeProximity]

    profile_alias = "BodyProximity"

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("manual_center_eye", "manual_torso", "manual_tail_label"))
        return upstream

    @cached_property
    def center_eye_proximity(self) -> ComputeProximity:
        return (
            self.manual_center_eye
            if self.manual_center_eye
            else ComputeProximity(
                perimeter=self.perimeter,
                perimeter_border_normal_pixels=self.maximum_distance_pixels,
                inside_perimeter_border=self.reader[self.center_eye_label],
                manual_video=self.video,
            )
        )

    @cached_property
    def torso_proximity(self) -> ComputeProximity:
        return (
            self.manual_torso
            if self.manual_torso
            else ComputeProximity(
                perimeter=self.perimeter,
                perimeter_border_normal_pixels=self.maximum_distance_pixels,
                inside_perimeter_border=self.reader[self.torso_label],
                manual_video=self.video,
            )
        )

    @cached_property
    def base_tail_proximity(self) -> ComputeProximity:
        return (
            self.manual_tail_label
            if self.manual_tail_label
            else ComputeProximity(
                perimeter=self.perimeter,
                perimeter_border_normal_pixels=self.maximum_distance_pixels,
                inside_perimeter_border=self.reader[self.base_tail_label],
                manual_video=self.video,
            )
        )

    @property
    def result(self) -> NDArrayBool:
        return (
            plural_node_tolerance_model(
                self.center_eye_proximity.result,
                self.torso_proximity.result,
                self.base_tail_proximity.result,
                fps=self.fps,
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
                self.profile_alias: self.result,
            }
        )

    def plot(self) -> None:
        fig, axes = self.video.subplots(ncols=4, nrows=1)

        self.center_eye_proximity.plot(axes[0], self.video)
        self.torso_proximity.plot(axes[1], self.video)
        self.base_tail_proximity.plot(axes[2], self.video)

        self.plot_result(axes[3])
