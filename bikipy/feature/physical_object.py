from functools import cached_property, reduce
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import validator, root_validator

from bikipy import MATPLOTLIB_SCATTER_ALPHA
from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.core.base_class import BaseBikipyInspectMixin
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64

from bikipy.core.typing import TrialId
from bikipy.core.video import (
    VideoMetadata,
    VideoMetadataMixin,
    convert_meters_to_pixels,
)
from bikipy.feature.attention.proximity import (
    proximity_filter,
)
from bikipy.feature.tolerance.single import single_node_tolerance_filter
from bikipy.perimeter.base import SinglePerimeter, PerimeterSet
from bikipy.reader.base import Reader
from bikipy.utils.collection_utils import generic_multi_indexer
from bikipy.utils.plotting import generic_inspection_finalization, InspectArg

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

    @cached_property
    def not_observing(self) -> NDArrayFp64:
        return ~self.attention_observance_boolean_index

    @cached_property
    def attention_filtered_seconds_observing(self) -> float:
        return np.sum(self.attention_observance_boolean_index) / self.video.fps

    @cached_property
    def raw_seconds_observing(self) -> float:
        return np.sum(self.logical_location_and_gaze) / self.video.fps

    @cached_property
    def filtered_raw_observation_ratio(self) -> float:
        return self.attention_filtered_seconds_observing / self.raw_seconds_observing

    @cached_property
    def distance_from_per_frame(self) -> NDArrayFp64:
        return np.linalg.norm(self._gaze_travel_direction_point - self.perimeter.centroid_meters, axis=1)

    def inspect_attention(self):
        self._exporting_figure = True

        gaze_travel_direction_point = (
            convert_meters_to_pixels(self._gaze_travel_direction_point, self.video)
            if self._inspect_pixels
            else self._gaze_travel_direction_point
        )

        self.attention_axes[1][0].scatter(
            *gaze_travel_direction_point[self.logical_location_and_gaze].T, alpha=MATPLOTLIB_SCATTER_ALPHA, marker="x"
        )

        self.attention_axes[1][1].scatter(
            *gaze_travel_direction_point[self.attention_observance_boolean_index].T,
            alpha=MATPLOTLIB_SCATTER_ALPHA,
            marker="x",
        )

        if isinstance(self.inspect_arg, Path):
            name = f"{self.inspect_arg.stem}_{self.label}.svg"
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
            figsize=(self.video.horizontal_resolution / 50.0, self.video.vertical_resolution / 50.0),
            constrained_layout=True,
        )

        for row_ax in self._axes:
            for col_ax in row_ax:
                if self.video.frame is not None:
                    # This will be done twice for row 0, as the perimeter plotter also plots video frame.
                    col_ax.autoscale(enable=True)
                    col_ax.imshow(self.video.frame)
                    col_ax.invert_yaxis()

                col_ax.set_aspect("equal", adjustable="box")

        self.attention_axes[1][0].set_title("proximity_filtered & gaze_filtered")
        self.attention_axes[1][1].set_title("Observation")

        self._fig.suptitle("Observation cumulative filtration analysis", fontsize=30)

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


class PhysicalObjectSet(VideoMetadataMixin):
    """
    The physical object set provides useful methods that compute relational features of physical-objects.
    Some methods are designed specifically for sets with a specific number of objects, while others are general.
    """

    physical_objects: tuple[PhysicalObject, ...] = ...

    overlapping_frame_to_total_frame_warning_ratio: ClassVar[float] = 0.05

    @cached_property
    def __len__(self) -> int:
        return len(self.physical_objects)

    def __getitem__(self, item):
        return self.label_to_physical_object[item]

    @cached_property
    def _video(self):
        return reduce(VideoMetadata.join, (physical_object.video for physical_object in self.physical_objects))

    @classmethod
    def from_perimeter(cls, *perimeters, **kwargs):
        return cls(physical_objects=tuple(PhysicalObject(perimeter=perimeter, **kwargs) for perimeter in perimeters))

    @classmethod
    def from_perimeter_set(cls, perimeter_set: PerimeterSet):
        assert not perimeter_set.restricted_perimeters
        return cls.from_perimeter(*perimeter_set.perimeters)

    @classmethod
    def from_bikipy_trial(cls, perimeters: Iterable[SinglePerimeter], trial_class):
        return cls(
            physical_objects=tuple(
                PhysicalObject(perimeter=perimeter, **trial_class.physical_object_keyword_arguments)
                for perimeter in perimeters
            )
        )

    @cached_property
    def frames(self) -> int:
        return len(self._first_object)

    @cached_property
    def observing_per_frame(self):
        return np.logical_or.reduce(
            [physical_object.attention_observance_boolean_index for physical_object in self.physical_objects]
        )

    @cached_property
    def not_observing_per_frame(self):
        return ~self.observing_per_frame

    @cached_property
    def seconds_observing(self):
        return np.sum(self.observing_per_frame) / self.video.fps

    @cached_property
    def seconds_not_observing(self):
        return np.sum(self.not_observing_per_frame) / self.video.fps

    @cached_property
    def object_specific_observation(self) -> dict[Any, int]:
        return {
            physical_object.label: physical_object.attention_filtered_seconds_observing
            for physical_object in self.physical_objects
        }

    @property
    def feature_summary(self) -> pd.Series:
        return pd.Series((self.seconds_observing, *self.object_specific_observation.values()))

    @cached_property
    def observation_sequence(self):
        overlapping_frames = 0

        result = np.zeros(len(self._first_object), dtype=np.uint8)
        for label, physical_object in self.label_to_physical_object.items():
            current_boolean_index = physical_object.attention_observance_boolean_index

            overlapping_frames += np.sum(current_boolean_index & result)

            result[current_boolean_index] = int(label.split("_")[1])  # TODO: Revert

        if overlapping_frames:
            ratio = overlapping_frames / self.frames
            logger.info(f"{overlapping_frames} out of {self.frames} overlap; ratio {overlapping_frames / self.frames}")
            if ratio > self.overlapping_frame_to_total_frame_warning_ratio:
                logger.warning(
                    f"Ratio is above the warning ratio, {self.overlapping_frame_to_total_frame_warning_ratio}:\n"
                    "This is a soft warning; do not be alarmed. Two or more objects have a temporal "
                    "overlap in their observation_sequence. This warning can be  ignored in most instances. "
                    "You have been warned that this observation_sequence data "
                    "MIGHT be empirically wrong. You may re-annotate the perimeters "
                    "of the objects for improved results"
                )

        return result

    @cached_property
    def reduced_observation_sequence(self):
        return np.array(
            reduce_repeating_sequences(self.observation_sequence, frame_tolerance=round(self.video.fps / 0.35))
        )

    @cached_property
    def physical_object_id_to_observation_instances(self):
        return {
            label: count
            for label, count in np.dstack(np.unique(self.reduced_observation_sequence, return_counts=True))[0]
        }

    @cached_property
    def sum_of_observation_instances(self):
        return sum(self.physical_object_id_to_observation_instances.values())

    @cached_property
    def object_bias_score(self) -> dict:
        if not self.seconds_observing:
            return self._label_to_zero
        return {
            label: 100.0 * physical_object.attention_filtered_seconds_observing / self.seconds_observing
            for label, physical_object in self.label_to_physical_object.items()
        }

    @cached_property
    def absolute_pair_discrimination(self) -> float:
        if len(self) != 2:
            msg = f"novel constant discrimination requires that the object number of the set is 2, not {len(self)}"
            raise AttributeError(msg)
        return abs(
            self.physical_objects[0].attention_filtered_seconds_observing
            - self.physical_objects[1].attention_filtered_seconds_observing
        )

    @cached_property
    def label_to_physical_object(self) -> dict:
        return {physical_object.label: physical_object for physical_object in self.physical_objects}

    def plot(self, ax: Any = None):
        if not ax:
            fix, ax = plt.subplots()
        for physical_objects in self.physical_objects:
            ax = physical_objects.perimeter.plot(ax=ax)

        return ax

    @cached_property
    def _label_to_zero(self) -> dict:
        return {label: 0.0 for label in self.label_to_physical_object}

    @cached_property
    def _first_object(self):
        return self.physical_objects[0]

    @validator("physical_objects", pre=True)
    def more_than_one_object(cls, value):
        if len(value) <= 1:
            msg = "Number of physical_objects in a set needs to be more than one"
            raise ValueError(msg)
        return value

    @validator("physical_objects", pre=True)
    def identical_frames(cls, value):
        if any(len(value[0]) != len(physical_object) for physical_object in value[1:]):
            msg = (
                f"The number of frames differ across physical objects:\n"
                f"{', '.join(str(len(physical_object)) for physical_object in value)}"
            )
            raise AttributeError(msg)
        return value

    @validator("physical_objects", pre=True)
    def identical_fps(cls, value):
        if any(value[0].video.fps != physical_object.video.fps for physical_object in value[1:]):
            msg = (
                f"Frames per second differ across physical objects:\n"
                f"{', '.join((physical_object.video.fps for physical_object in value))}"
            )
            raise AttributeError(msg)
        return value
