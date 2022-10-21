from functools import cached_property, reduce
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import root_validator, validator
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64

from bikipy import runtime_settings
from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.core.base_class import BaseBikipyInspectMixin
from bikipy.core.typing import TrialId
from bikipy.core.video import (
    VideoMetadata,
    VideoMetadataMixin,
    prepare_data_for_plotting,
)
from bikipy.feature.attention.proximity import proximity_filter
from bikipy.feature.tolerance.single import single_node_tolerance_filter
from bikipy.perimeter.base import PerimeterSet, SinglePerimeter
from bikipy.reader.base import Reader
from bikipy.utils.collection_utils import generic_multi_indexer
from bikipy.utils.image import axis_frame_imshow
from bikipy.utils.plotting import InspectArg, generic_inspection_finalization

logger = getLogger(__name__)


class PhysicalObject(BaseBikipyInspectMixin):
    """
    The physical object is a triadic abstraction of Reader, Perimeter and Trial. This abstraction allows
    us to define methods that require the respective attributes, think of it as a union between the classes!
    """

    perimeter: SinglePerimeter = ...
    reader: Reader = ...

    # Proximity fields
    perimeter_border_normal_pixels: float | NDArrayFp64 = ...
    outside_perimeter_point_label: str | None

    # Gaze fields
    gaze_start_point_label: str = ...
    gaze_travel_direction_point_label: str = ...
    maximum_radians_inter_gaze_perimeter: float = ...

    # Tolerance fields
    minimum_seconds_attention: float = ...
    maximum_seconds_distraction: float = ...

    # Inspection fields
    trial_obj_label: Optional[TrialId]
    _fig: Any = None
    _axes: Any = None
    _exporting_figure: bool = False

    category = "physical_object"

    @root_validator
    def outside_perimeter_point_label_only_when_perimeter_is_impenetrable(cls, values):
        if not values["outside_perimeter_point_label"] and not values["perimeter"].impenetrable:
            msg = (
                f"Perimeter {values['perimeter'].label}: Physical object perimeter must be impenetrable "
                f"if outside_perimeter_point_label is set to None"
            )
            raise AttributeError(msg)
        return values

    def __len__(self) -> int:
        return self.reader.frames

    @cached_property
    def video(self) -> VideoMetadata:
        return VideoMetadata.join(self.perimeter.video, self.reader.video, ignore_incongruity=True)

    @property
    def label(self):
        return self.perimeter.label

    @cached_property
    def proximity_seconds(self) -> float:
        return np.sum(self.attention_proximity_boolean_index) / self.video.fps

    @cached_property
    def proximity_and_gaze_seconds(self) -> float:
        return np.sum(self.logical_location_and_gaze) / self.video.fps

    @cached_property
    def tolerance_filtered_proximity_and_gaze_seconds(self) -> float:
        return np.sum(self.attention_observance_boolean_index) / self.video.fps

    @cached_property
    def tolerance_vs_unfiltered_ratio(self) -> float:
        return self.tolerance_filtered_proximity_and_gaze_seconds / self.proximity_and_gaze_seconds

    @cached_property
    def attention_proximity_boolean_index(self) -> NDArrayBool:
        return proximity_filter(
            self.perimeter,
            self._gaze_travel_direction_point,
            self._outside_perimeter_point if self.outside_perimeter_point_label else self._gaze_start_point,
            self.perimeter_border_normal_pixels,
            manual_ax=self.attention_axes[0][0] if self.inspect_arg else None,
            **self._global_attention_kwargs,
        )

    @cached_property
    def attention_gaze_boolean_index(self) -> NDArrayBool:
        return self.perimeter.gaze_direction_filter(
            self._gaze_travel_direction_point,
            self._gaze_start_point,
            self.maximum_radians_inter_gaze_perimeter,
            manual_ax=self.attention_axes[0][1] if self.inspect_arg else None,
            **self._global_attention_kwargs,
        )

    @cached_property
    def logical_location_and_gaze(self) -> NDArrayFp64:
        return self.attention_proximity_boolean_index & self.attention_gaze_boolean_index

    @cached_property
    def attention_observance_boolean_index(self) -> NDArrayBool:
        result = single_node_tolerance_filter(
            self.logical_location_and_gaze,
            self.video.fps,
            self.minimum_seconds_attention,
            self.maximum_seconds_distraction,
        )

        if self.inspect_arg and not self._exporting_figure:
            self.inspect_attention()

        return result

    def inspect_attention(self):
        self._exporting_figure = True

        gaze_travel_direction_point = prepare_data_for_plotting(
            self._gaze_travel_direction_point, self._inspect_pixels, self.video
        )

        self.attention_axes[1][0].scatter(
            *gaze_travel_direction_point[self.logical_location_and_gaze].T,
            # alpha=runtime_settings.matplotlib_scatter_alpha,
            marker="x",
            color="b",
        )

        self.attention_axes[1][1].scatter(
            *gaze_travel_direction_point[self.attention_observance_boolean_index].T,
            # alpha=runtime_settings.matplotlib_scatter_alpha,
            marker="x",
            color="b",
        )

        # plt.tight_layout(pad=10)

        if isinstance(self.inspect_arg, Path):
            name = f"{self.inspect_arg.stem}_{self.label}.jpg"
            if self.trial_obj_label:
                name = f"{self.trial_obj_label}_{name}"
            generic_inspection_finalization(self.class_inspect_arg / name)
        else:
            generic_inspection_finalization(self.class_inspect_arg)

    @property
    def attention_fig(self):
        if self._fig is not None:
            return self._fig
        self._init_matplotlib()
        return self._fig

    @property
    def attention_axes(self):
        if self._axes is not None:
            return self._axes
        self._init_matplotlib()
        return self._axes

    def _init_matplotlib(self):
        self._fig, self._axes = plt.subplots(
            nrows=2,
            ncols=2,
            constrained_layout=True,
            figsize=(
                self.video.upscaled_video.horizontal_resolution * 0.1,
                self.video.upscaled_video.vertical_resolution * 0.1,
            ),
        )

        if self.video.frame is None:
            for row_ax in self._axes:
                for col_ax in row_ax:
                    col_ax.set_aspect("equal", adjustable="box")
        else:
            # axes row 1 will be targeted by analysis inspect function, no need to do that here
            for ax in (self.attention_axes[1][0], self.attention_axes[1][1]):
                axis_frame_imshow(ax, self.video.upscaled_video.greyscale_frame)
                self.video.upscaled_video.ax_ticks_metric_to_pixel(ax)

        self.attention_axes[1][0].set_title(
            "proximity_filtered & gaze_filtered", fontsize=self.video.upscaled_video.plotting_title_font_size
        )
        self.attention_axes[1][1].set_title("Observation", fontsize=self.video.upscaled_video.plotting_title_font_size)

        self._fig.suptitle(
            "Observation cumulative filtration analysis",
            fontsize=self.video.upscaled_video.plotting_title_font_size * 1.1,
        )

    @cached_property
    def _inspect_pixels(self) -> bool:
        return self.video.frame is not None

    @cached_property
    def _global_attention_kwargs(self) -> dict[str, Any]:
        return {"inspect_video": self.video, "inspect_pixels": self._inspect_pixels}

    @property
    def _outside_perimeter_point(self) -> NDArrayFp64:
        return self.reader[self.outside_perimeter_point_label]

    @property
    def _gaze_start_point(self) -> NDArrayFp64:
        return self.reader[self.gaze_start_point_label]

    @property
    def _gaze_travel_direction_point(self) -> NDArrayFp64:
        return self.reader[self.gaze_travel_direction_point_label]
