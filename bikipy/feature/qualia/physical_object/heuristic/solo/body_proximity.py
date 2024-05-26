from functools import cached_property
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import computed_field
from pydantic_numpy.typing import Np1DArrayBool

from bikipy.feature.qualia.axioms.proximity import ComputeProximity
from bikipy.feature.qualia.physical_object.heuristic.mixin import ProximityMixin
from bikipy.feature.qualia.physical_object.heuristic.solo.abc import (
    AbstractSoloHeuristic,
)
from bikipy.perimeter.base import BasePerimeter


class BodyProximityHeuristic(AbstractSoloHeuristic, ProximityMixin):
    center_ear_label: str | None = "center_ear"
    tail_base_label: str | None = "tail_base"

    manual_center_ear: Optional[ComputeProximity] = None
    manual_tail_base: Optional[ComputeProximity] = None

    heuristic_alias = "BodyProximity"

    @classmethod
    @property
    def fields_to_exclude_from_single_schema(cls) -> set[str]:
        result = super().fields_to_exclude_from_single_schema
        result.update(("manual_center_ear", "manual_torso", "manual_tail_base"))
        return result

    @computed_field  # type: ignore[misc]
    @cached_property
    def center_ear_proximity(self) -> ComputeProximity | None:
        if self.manual_center_ear is not None:
            return self.manual_center_ear

        if not self.center_ear_label:
            return None

        return ComputeProximity(
            perimeter=self.perimeter,
            maximum_distance=self.maximum_distance_pixels,
            inside_perimeter_border=self.reader[self.center_ear_label],
            label="Center ear",
            manual_video=self.video,
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def tail_base_proximity(self) -> ComputeProximity | None:
        if self.manual_tail_base is not None:
            return self.manual_tail_base

        if not self.tail_base_label:
            return None

        return ComputeProximity(
            perimeter=self.perimeter,
            maximum_distance=self.maximum_distance_pixels,
            inside_perimeter_border=self.reader[self.tail_base_label],
            label="Tail base",
            manual_video=self.video,
        )

    @computed_field  # type: ignore[misc]
    @property
    def solo_result(self) -> Np1DArrayBool:
        return np.logical_or.reduce(
            [node.result for node in (self.center_ear_proximity, self.tail_base_proximity) if node is not None]
        )

    @computed_field  # type: ignore[misc]
    @property
    def label_to_proximity_boolean(self) -> dict[str, Np1DArrayBool]:
        return {
            self.center_ear_label: self.center_ear_proximity.result,
            self.tail_base_label: self.tail_base_proximity.result,
        }

    @computed_field  # type: ignore[misc]
    @property
    def perimeter_to_boolean_index(self) -> dict[BasePerimeter, Np1DArrayBool]:
        return self.center_ear_proximity.video_gen_merge_perimeter_to_boolean_index(self.tail_base_proximity)

    @computed_field  # type: ignore[misc]
    @property
    def summary_series(self) -> pd.Series:
        data = {}
        label = self.perimeter.label.capitalize()

        if self.center_ear_proximity:
            data[f"{label}CenterEarProximity"] = self.center_ear_proximity.result_seconds

        if self.tail_base_proximity:
            data[f"{label}BaseTailProximity"] = self.tail_base_proximity.result_seconds

        data[f"{label}{self.heuristic_alias}Result"] = self.video.boolean_array_to_seconds(self.result)

        return pd.Series(data)

    def plot(self) -> plt.Figure:
        fig, axes = self.perimeter.subplots(
            ncols=sum((bool(self.center_ear_proximity), bool(self.tail_base_proximity))) + 1,
            nrows=1,
            title=self.heuristic_alias,
        )

        ax_idx = 0

        if self.center_ear_proximity:
            axes[ax_idx].set_title("Center ear")
            self.center_ear_proximity.plot(axes[ax_idx], self.video)
            ax_idx += 1

        if self.tail_base_proximity:
            axes[ax_idx].set_title("Tail base")
            self.tail_base_proximity.plot(axes[ax_idx], self.video)
            ax_idx += 1

        self.plot_result(axes[ax_idx], self.center_ear_label or self.tail_base_label)

        return fig
