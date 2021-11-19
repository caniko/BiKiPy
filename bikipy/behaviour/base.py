import os
import re
from abc import ABC, abstractproperty
from copy import copy
from functools import cached_property
from logging import getLogger
from pathlib import Path, PurePath
from typing import Any, Iterable, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, DirectoryPath, Field, FilePath
from tqdm import tqdm

from bikipy._base_class import BikipyBase, VideoMetaDataMixin
from bikipy.feature.motion import Motion, motion_2d_multi_indexer
from bikipy.reader.deeplabcut import DeepLabCutReader
from bikipy.utils.store import RangeDict
from bikipy.utils.typing import NDArray

logger = getLogger(__name__)


LABEL_VS_DATA_READER = {"deeplabcut": DeepLabCutReader}


class Behaviour(BikipyBase, ABC):
    fps: Union[float, int, None] = None
    recording_resolution: Optional[NDArray[Literal["np.int16"]]] = None
    metric_resolution: Union[NDArray, float, None] = None
    manual_units_per_pixel: Optional[float] = None
    data_import_kwargs: Optional[dict] = None
    data_format_label: Literal["deeplabcut"] = "deeplabcut"

    _live: bool = False

    @property
    def horizontal_resolution(self):
        try:
            return self.recording_resolution[0]
        except TypeError:
            msg = (
                "recording_resolution needs to be defined to for "
                "the acquisition of horizontal_resolution"
            )
            raise AttributeError(msg)

    @property
    def vertical_resolution(self):
        try:
            return self.recording_resolution[1]
        except TypeError:
            msg = (
                "recording_resolution needs to be defined to for "
                "the acquisition of vertical_resolution"
            )
            raise AttributeError(msg)

    @property
    def units_per_pixel(self):
        return self.manual_units_per_pixel or self.computed_units_per_pixel

    @cached_property
    def computed_units_per_pixel(self):
        if not np.any(self.metric_resolution):
            msg = "metric_resolution attribute needs to be defined to compute units_per_pixel"
            raise AttributeError(msg)
        if isinstance(self.metric_resolution, (float, int)):
            return self.metric_resolution / np.mean(self.recording_resolution)
        else:
            return np.array(self.metric_resolution) / self.recording_resolution


class BaseExperiment(Behaviour, ABC):
    trial_class: Any = None
    trial_id_vs_trial_class: Optional[dict] = None
    point_label_for_motion_features: Optional[str] = None
    trial_id_vs_keyword_arguments: Optional[dict] = None
    trial_id_range_vs_keyword_arguments: Optional[RangeDict] = None
    common_trial_keyword_arguments: Optional[dict] = None
    inspection_figure_save: Union[DirectoryPath, bool] = False

    _enable_process_pooling = True
    _deeplabcut_trial_id_finder = re.compile(r"\d+")

    def __getitem__(self, item: int):
        return self.trial_id_vs_trial_object[item]

    @cached_property
    def feature_summary_frame(self) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
        """
        Experiment classes must implement this property for the generation of
        summary frames
        """
        if self.trial_class:
            # Only one DataFrame schema
            return pd.DataFrame(
                (trial.feature_summary_row for trial in self.trial_objects),
                columns=self._feature_summary_column,
                index=self._frame_index,
            )
        elif self.trial_id_vs_trial_class:
            return {
                trial_class_name: pd.DataFrame(
                    (trial.feature_summary_row for trial in trial_objects),
                    columns=self._feature_summary_column,
                    index=self._frame_index,
                )
                for trial_class_name, trial_objects in self.trial_class_name_vs_trial_objects.items()
            }

    def trial_keyword_arguments(self, trial_id: int) -> dict:
        """
        Function useful for customizing initiation parameters for trial objects

        :param trial_id: Respective trial ID
        :return:
        """
        result = (
            copy(self.common_trial_keyword_arguments)
            if self.common_trial_keyword_arguments
            else {}
        )
        result["data_format_label"] = self.data_format_label

        if self.trial_id_vs_keyword_arguments:
            result.update(self.trial_id_vs_keyword_arguments[trial_id])
        if self.trial_id_range_vs_keyword_arguments:
            result.update(self.trial_id_range_vs_keyword_arguments[trial_id])
        if self.data_import_kwargs:
            result["data_import_kwargs"] = self.data_import_kwargs
        return result

    @cached_property
    def trial_objects(self) -> list:
        if self.trial_class:
            return [
                self.trial_class(**self.trial_keyword_arguments(trial_id))
                for trial_id in self._trial_id_key_view
            ]
        elif self.trial_id_vs_trial_class:
            return [
                self.trial_id_vs_trial_class[trial_id](
                    **self.trial_keyword_arguments(trial_id)
                )
                for trial_id in self._trial_id_key_view
            ]
        else:
            msg = (
                "Either trial_class or trial_id_vs_trial_class have to be "
                "exclusively defined"
            )
            raise AttributeError(msg)

    @cached_property
    def _trial_class_vs_trial_ids(self):
        if not self.trial_id_vs_trial_class:
            msg = (
                "This experiment object has no trial_id_vs_trial_class, "
                "this attribute is reserved for experiments with "
                "several trial classes"
            )
            raise AttributeError(msg)

        result = {}
        for trial_class, trial_id in self.trial_id_vs_trial_class:
            if trial_class in result:
                result[trial_class].append(trial_id)
            else:
                result[trial_class] = [trial_id]

        return result

    @cached_property
    def trial_class_name_vs_trial_ids(self):
        return {
            trial_class.__name__: trial_ids
            for trial_class, trial_ids in self._trial_class_vs_trial_ids
        }

    @cached_property
    def trial_class_name_vs_trial_objects(self):
        result = {}
        for trial_class_name, trial_ids in self.trial_class_name_vs_trial_ids.items():
            result[trial_class_name] = [
                self.trial_id_vs_trial_object[trial_id] for trial_id in trial_ids
            ]
        return result

    @cached_property
    def trial_id_vs_trial_object(self) -> dict:
        return {trial.int_id: trial for trial in self.trial_objects}

    @cached_property
    def animal_id_vs_trial_objects(self) -> dict:
        result = {}
        for trial in self.trial_objects:
            if trial.animal_id in result:
                result[trial.animal_id].append(trial)
            else:
                result[trial.animal_id] = [trial]
        return result

    @property
    def inspect(self):
        if isinstance(self.inspection_figure_save, bool):
            return self.inspection_figure_save
        elif isinstance(self.inspection_figure_save, PurePath):
            if not self.inspection_figure_save.exists():
                os.makedirs(self.inspection_figure_save)
            return self.inspection_figure_save / f"experiment_{self.timestamp}_inspect"
        else:
            raise AttributeError()

    @property
    def _trial_id_key_view(self):
        if self.trial_id_vs_keyword_arguments:
            return self.trial_id_vs_keyword_arguments.keys()
        if self.trial_id_range_vs_keyword_arguments:
            return self.trial_id_range_vs_keyword_arguments.keys()

    @property
    def trial_id_tuple(self) -> tuple:
        return tuple(self._trial_id_key_view)

    @cached_property
    def number_of_trials(self):
        return len(self._trial_id_key_view)

    @property
    def trial_id_data_tqdm(self):
        return tqdm(
            ((trial_id, self[trial_id]) for trial_id in self._trial_id_key_view),
            total=self.number_of_trials,
        )

    @staticmethod
    def _feature_2d_multi_indexer(feature: str, category):
        return tuple([(feature, category) for category in category])

    @property
    def _frame_index(self):
        return pd.Series(self._trial_id_key_view, name="Test ID", dtype=np.int16)

    @cached_property
    def _trial_id_vs_animal_id_frame(self):
        if not any(not trial.animal_id for trial in self.trial_objects):
            return pd.DataFrame(
                (trial.animal_id for trial in self.trial_objects),
                columns=("Animal ID",),
                index=self._frame_index,
            )

    @cached_property
    def motion_summary_frame(self):
        result = pd.DataFrame(
            (trial.motion.to_list for trial in self.trial_objects),
            columns=motion_2d_multi_indexer("All"),
            index=self._frame_index,
        )
        if self._trial_id_vs_animal_id_frame:
            return pd.concat((self._trial_id_vs_animal_id_frame, result), axis=1)
        return result


class BaseTrial(Behaviour, VideoMetaDataMixin, ABC):
    coordinate_data_path: FilePath = Field(
        description="Path to file storing coordinate data"
    )
    animal_id: Optional[int] = Field(
        None, description="The ID of the animal in the trial"
    )
    point_label_for_motion_features: Optional[str] = Field(
        description="Label of the node that will be used to track general animal movement"
    )
    rigid_nodes_freezing: Optional[Sequence[Union[str, int]]] = Field(
        None,
        description="Nodes that should remain during freeze/immobility, most often due to fear.",
    )
    inspection_figure_save: Union[DirectoryPath, bool] = Field(
        False, description="Path to save figures for inspection of results"
    )
    inspect_image: Optional[FilePath] = Field(
        None, description="Image used for inspection"
    )
    # Variables for trials with zones, see doc for more info.
    perimeters: Optional[Sequence] = None
    trial_start_perimeter: Optional[str] = None

    _category = "trial"

    _trial_sequence_index = None
    _trial_label = None
    _second_tolerance = 0.35

    _trial_has_feature_frame = True
    _feature_summary_column = None

    @property
    def feature_summary_row(self) -> list:
        """
        Trial classes must implement this property for the generation of
        summary frames
        """
        if self._trial_has_feature_frame:
            logger.warning(
                "The base version of feature_summary_row property is being "
                "used. Note that this will yield an empty summary frame. "
                "This property needs to be replaced"
            )
        return []

    @staticmethod
    def _get_reader(coordinate_data_format):
        try:
            return LABEL_VS_DATA_READER[coordinate_data_format]
        except KeyError as e:
            msg = (
                f"{coordinate_data_format} as a format for data ingestion has "
                f"no implementation"
            )
            raise NotImplemented(msg) from e

    @cached_property
    def reader(self):
        return self._get_reader(self.data_format_label)(
            self.coordinate_data_path,
            **self._video_metadata_dict_manual_format,
            **self.data_import_kwargs,
        )

    @property
    def coordinates_per_frame(self) -> np.ndarray:
        return self.reader[self.point_label_for_motion_features]

    @cached_property
    def number_of_frames(self) -> int:
        return len(self.coordinates_per_frame)

    @cached_property
    def experiment_seconds(self) -> int:
        return self.coordinates_per_frame.shape[0] / self.fps

    @property
    def inspect_image_path(self):
        if isinstance(self.inspection_figure_save, str) or isinstance(
            self.inspection_figure_save, PurePath
        ):
            return Path(self.inspection_figure_save) / self.best_id
        return self.inspection_figure_save  # return the bool in any case

    @cached_property
    def info(self):
        return [self._trial_label] if self._trial_label else []

    @cached_property
    def recording_center_pixel(self) -> np.ndarray:
        return self.recording_resolution / 2.0

    @cached_property
    def motion(self):
        return Motion(self.coordinates_per_frame, self.units_per_pixel, self.fps)

    # Perimeter

    def detect_confined_perimeter(self, coordinate: np.array):
        """
        This function is used to determine current location of subject.

        :param coordinate:
        :return:
        """

        coordinate = np.expand_dims(coordinate, 0)
        for label, perimeter in self.int_id_vs_perimeter.items():
            if perimeter.coordinate_confinement_boolean_index(coordinate):
                logger.info(f"Location: {label}, {coordinate}")
                return label
        logger.debug(f"Location could not be determined, {coordinate}")

    @cached_property
    def int_id_vs_perimeter(self):
        self._validate_perimeters_object()
        return {
            i: perimeter
            for i, perimeter in enumerate(self.perimeters.values(), start=1)
        }

    def _validate_perimeters_object(self):
        if not self.perimeters:
            msg = "perimeters is not defined as an object variable, which is required for int_id_vs_perimeter"
            raise AttributeError(msg)

    @cached_property
    def _perimeter_label_vs_int_id(self):
        self._validate_perimeters_object()
        return {label: i for i, label in enumerate(self.perimeters, start=1)}

    @cached_property
    def _int_id_vs_perimeter_label(self):
        self._validate_perimeters_object()
        return {i: label for i, label in enumerate(self.perimeters, start=1)}

    @property
    def _start_int_id(self) -> int:
        return self._perimeter_label_vs_int_id[self.trial_start_perimeter]

    def _perimeter_label_sequence_to_int_id(self, label_sequence: Iterable) -> tuple:
        return tuple(self._perimeter_label_vs_int_id[label] for label in label_sequence)

    # Miscellaneous

    @cached_property
    def perimeter_border_normal_pixel_magnitude(self):
        return self.perimeter_border_normal_metric_magnitude / np.mean(
            self.units_per_pixel
        )

    @cached_property
    def _physical_object_keyword_arguments(self):
        try:
            return {
                "reader": self.reader,
                "gaze_travel_direction_point_label": self.gaze_travel_direction_point_label,
                "gaze_start_point_label": self.gaze_start_point_label,
                "fps": self.fps,
                "perimeter_border_normal_pixel_magnitude": self.perimeter_border_normal_pixel_magnitude,
                "maximum_radians_inter_gaze_perimeter": self.maximum_radians_inter_gaze_perimeter,
                "minimum_seconds_attention": self.minimum_seconds_attention,
                "inspect": self.inspection_figure_save
            }
        except AttributeError as e:
            msg = "The class does not support instancing PhysicalObject"
            raise NotImplementedError(msg) from e

    @cached_property
    def _zeros_based_on_frame_length(self) -> np.ndarray:
        return np.zeros(self.number_of_frames, dtype=np.uint8)

    @cached_property
    def _frame_tolerance(self) -> int:
        return round(self._second_tolerance * self.fps)
