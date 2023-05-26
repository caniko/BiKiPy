from functools import cached_property
from typing import Optional

import numpy as np
import pandas as pd

from bikipy.feature.qualia.axioms.proximity import ComputeProximity
from bikipy.feature.qualia.physical_object.heuristic.abc import (
    AbstractQualiaHeuristic,
    ProximityMixin,
)


class BodyProximityHeuristic(AbstractQualiaHeuristic, ProximityMixin):
    center_eye_label: str | None = "center_eye"
    torso_label: str | None = "torso"
    tail_base_label: str | None = "tail_base"

    manual_center_eye: Optional[ComputeProximity]
    manual_torso: Optional[ComputeProximity]
    manual_tail_base: Optional[ComputeProximity]

    heuristic_alias = "BodyProximity"

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("manual_center_eye", "manual_torso", "manual_tail_base"))
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
            should_be_outside_perimeter_border=self.reader[self.torso_label] if self.perimeter.impenetrable else None,
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
            should_be_outside_perimeter_border=self.reader[self.torso_label] if self.perimeter.impenetrable else None,
            label="Torso",
            manual_video=self.video,
        )

    @cached_property
    def tail_base_proximity(self) -> ComputeProximity | None:
        if self.manual_tail_base is not None:
            return self.manual_tail_base

        if not self.tail_base_label:
            return None

        return ComputeProximity(
            perimeter=self.perimeter,
            perimeter_border_normal_pixels=self.maximum_distance_pixels,
            should_be_inside_perimeter_border=self.reader[self.tail_base_label],
            should_be_outside_perimeter_border=self.reader[self.torso_label] if self.perimeter.impenetrable else None,
            label="Tail base",
            manual_video=self.video,
        )

    @cached_property
    def result(self) -> np.ndarray[bool, bool]:
        nodes = [
            node.result
            for node in (self.center_eye_proximity, self.torso_proximity, self.tail_base_proximity)
            if node is not None
        ]

        return np.logical_or.reduce(nodes)

    @property
    def summary_series(self) -> pd.Series:
        data = {}
        label = self.perimeter.label.capitalize()

        if self.center_eye_proximity:
            data[f"ObservingSecCenterEyeProximity{label}"] = self.center_eye_proximity.result_seconds

        if self.torso_proximity:
            data[f"ObservingSecTorsoProximity{label}"] = self.torso_proximity.result_seconds

        if self.tail_base_proximity:
            data[f"ObservingSecBaseTailProximity{label}"] = self.tail_base_proximity.result_seconds

        data[f"ObservingSec{self.heuristic_alias}Total{label}"] = self.boolean_array_to_seconds(self.result)

        return pd.Series(data)

    def plot(self) -> None:
        fig, axes = self.video.subplots(
            ncols=sum((bool(self.center_eye_proximity), bool(self.torso_proximity), bool(self.tail_base_proximity)))
            + 1,
            nrows=1,
        )

        ax_idx = 0

        if self.center_eye_proximity:
            axes[ax_idx].set_title("Center eye")
            self.center_eye_proximity.plot(axes[ax_idx], self.video)
            ax_idx += 1

        if self.torso_proximity:
            axes[ax_idx].set_title("Torso")
            self.torso_proximity.plot(axes[ax_idx], self.video)
            ax_idx += 1

        if self.tail_base_proximity:
            axes[ax_idx].set_title("Tail base")
            self.tail_base_proximity.plot(axes[ax_idx], self.video)
            ax_idx += 1

        self.plot_result(axes[ax_idx], self.torso_label or self.center_eye_label or self.tail_base_label)
