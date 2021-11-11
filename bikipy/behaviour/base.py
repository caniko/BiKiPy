import os
import re
from abc import ABC
from functools import cached_property, partial
from logging import getLogger
from pathlib import Path, PurePath
from typing import Iterable, Literal, Optional, Sequence, Union, Any

import cv2
import numpy as np
import pandas as pd
from pydantic import DirectoryPath, Field, FilePath
from tqdm import tqdm

from bikipy._base_class import BikipyBase
from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.feature.motion import Motion, displacement_by_frame, frozen_frames
from bikipy.math.point_in_polygon import points_in_parallelogram
from bikipy.reader.deeplabcut import DeepLabCutReader
from bikipy.utils.store import RangeDict
from bikipy.utils.typing import NDArray
from bikipy.utils.video import get_video_data

logger = getLogger(__name__)


LABEL_VS_DATA_READER = {"deeplabcut": DeepLabCutReader}


class Behaviour(BikipyBase, ABC):
    fps: Union[float, int, None] = None
    recording_resolution: Optional[NDArray[Literal["np.int16"]]] = None
    metric_resolution: Union[NDArray, float, None] = None
    manual_units_per_pixel: Optional[float] = None
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
    readers: list
    trial_id_vs_trial_class: dict
    trial_id_range_vs_trial_keyword_arguments: Optional[RangeDict] = None
    common_trial_keyword_arguments: Optional[dict] = None
    point_label_for_motion_features: Optional[str] = None
    inspection_figure_save: Union[DirectoryPath, bool] = False

    _enable_process_pooling = True
    _deeplabcut_trial_id_finder = re.compile(r"\d+")

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

    @classmethod
    def from_deeplabcut_data(
        cls,
        coordinate_data_paths: Sequence,
        experiment_kwargs: dict,
        data_import_kwargs: Optional[dict] = None,
    ):
        reader = cls._get_reader("deeplabcut")
        cls(
            readers=[
                reader(
                    df_path=data_path,
                    int_id=cls._deeplabcut_trial_id_finder.findall(
                        Path(data_path).stem
                    )[0],
                    **data_import_kwargs,
                )
                for data_path in coordinate_data_paths
            ],
            **experiment_kwargs,
        )

    def trial_keyword_arguments(self, trial_id: int):
        trial_meta = self[trial_id]
        generic_kwargs = {
            "int_id": trial_id,
            "reader": self.trial_id_vs_reader[trial_id],
            "video_path": trial_meta["video_path"],
            "metric_resolution": self.metric_resolution,
            "inspection_figure_save": self.inspection_figure_save,
        }
        try:
            generic_kwargs["units_per_pixel"] = self.units_per_pixel
        except TypeError:
            pass

        if "animal_id" in trial_meta:
            generic_kwargs["animal_id"] = trial_meta["animal_id"]

        if "inspect" in trial_meta:
            generic_kwargs["inspection_figure_save"] = trial_meta["inspect"]
            if "inspect_image" in trial_meta:
                generic_kwargs["inspect_image"] = trial_meta["inspect_image"]

        return generic_kwargs

    @cached_property
    def trial_id_vs_reader(self):
        return {reader.int_id: reader for reader in self.readers}

    @cached_property
    def trial_objects(self):
        result = []
        for reader in self.readers:
            trial_id = reader.int_id
            result.append(
                self.trial_id_vs_trial_class[trial_id](
                    **self.trial_keyword_arguments(trial_id)
                )
            )
        return result

    @cached_property
    def trial_id_vs_trial_object(self):
        return {trial.int_id: trial for trial in self.trial_objects}

    def __getitem__(self, item: int):
        return self.trial_id_vs_trial_object[item]

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
    def _trial_id_iterable(self):
        return self.trial_id_vs_reader.keys()

    @cached_property
    def number_of_trials(self):
        return len(self._trial_id_iterable)

    @property
    def trial_id_data_tqdm(self):
        return tqdm(
            ((trial_id, self[trial_id]) for trial_id in self._trial_id_iterable),
            total=self.number_of_trials,
        )

    @staticmethod
    def _motion_2d_multi_indexer(category: str):
        _category = str(category)
        return (
            (category, "Displacement"),
            (category, "Median_speed"),
            (category, "Median_acceleration"),
            (category, "Freezing time"),
        )

    @staticmethod
    def _feature_2d_multi_indexer(feature: str, category):
        return tuple([(feature, category) for category in category])

    @property
    def _frame_index(self):
        return pd.Series(self._trial_id_iterable, name="Test ID", dtype=np.int16)

    @cached_property
    def motion_summary_frame(self):
        return pd.DataFrame(
            (trial.motion.to_list for trial in self.trial_objects),
            columns=self._motion_2d_multi_indexer("All"),
            index=self._frame_index,
        )


class BaseTrial(Behaviour, ABC):
    reader: Any = Field(
        description="The coordinates of the subject across the frames in the video recording"
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
    video_path: Optional[FilePath] = Field(
        None, description="Path to trial video recording"
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

    class Config:
        fields = {"rigid_nodes_freezing_": "rigid_nodes_freezing"}

    def __init__(self, **data):
        super().__init__(**data)

        if self.video_path:
            (
                _frame,
                horizontal_resolution,
                vertical_resolution,
                self.fps,
            ) = get_video_data(self.video_path)
            self.recording_resolution = np.array(
                (horizontal_resolution, vertical_resolution), dtype=np.int16
            )
        elif self.recording_resolution and self.fps:
            pass
        elif self.inspect_image and self.fps:
            self.recording_resolution = cv2.imread(self.inspect_image).shape[:-1]
        else:
            msg = (
                "recording_resolution and fps could not be defined."
                "One of the following compbinations must be provided:\n"
                "\t1. Trial video path\n"
                "\t2. recording_resolution and frame per second (fps)\n"
                "\t3. inspect_image and frame per second (fps)"
            )
            raise ValueError(msg)

    @property
    def coordinates_per_frame(self):
        return self.reader[self.point_label_for_motion_features]

    @cached_property
    def number_of_frames(self):
        return len(self.coordinates_per_frame)

    @cached_property
    def experiment_seconds(self):
        return self.coordinates_per_frame.shape[0] / self.fps

    @cached_property
    def motion(self):
        return Motion(
            self.coordinates_per_frame,
            self.units_per_pixel,
            self.fps,
            label_vs_boolean_index={
                "quadrant_upper_left": self.quadrant_upper_left_boolean_index,
                "quadrant_upper_right": self.quadrant_upper_right_boolean_index,
                "quadrant_down_left": self.quadrant_down_left_boolean_index,
                "quadrant_down_right": self.quadrant_down_right_boolean_index,
            },
        )

    @property
    def _hash_key(self):
        return self.coordinates_per_frame

    @property
    def inspect_image_path(self):
        if isinstance(self.inspection_figure_save, str) or isinstance(
            self.inspection_figure_save, PurePath
        ):
            return Path(self.inspection_figure_save) / self.best_id
        return self.inspection_figure_save  # return the bool in any case

    @cached_property
    def _frame_tolerance(self):
        return round(self._second_tolerance * self.fps)

    @cached_property
    def frozen_boolean_index(self):
        if not self.rigid_nodes_freezing:
            msg = "rigid_nodes_freezing is not defined"
            raise AttributeError(msg)
        return frozen_frames(
            self.fps,
            [
                displacement_by_frame(reader, remove_tails=False) * self.units_per_pixel
                for reader in self.reader[self.rigid_nodes_freezing]
            ],
        )

    @cached_property
    def total_frozen_frames(self):
        return np.sum(self.frozen_boolean_index) / self.fps

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

    def _validate_perimeters_object(self):
        if not self.perimeters:
            msg = "perimeters is not defined as an object variable, which is required for int_id_vs_perimeter"
            raise AttributeError(msg)

    @cached_property
    def _perimeter_label_vs_int_id(self):
        self._validate_perimeters_object()
        return {label: i for i, label in enumerate(self.perimeters.keys(), start=1)}

    @cached_property
    def _int_id_vs_perimeter_label(self):
        self._validate_perimeters_object()
        return {i: label for i, label in enumerate(self.perimeters.keys(), start=1)}

    @cached_property
    def int_id_vs_perimeter(self):
        self._validate_perimeters_object()
        return {
            i: perimeter
            for i, perimeter in enumerate(self.perimeters.values(), start=1)
        }

    @property
    def _start_int_id(self):
        return self._perimeter_label_vs_int_id[self.trial_start_perimeter]

    def _perimeter_label_sequence_to_int_id(self, label_sequence: Iterable) -> tuple:
        return tuple(self._perimeter_label_vs_int_id[label] for label in label_sequence)

    @cached_property
    def info(self):
        return [self._trial_label] if self._trial_label else []

    @cached_property
    def recording_center_pixel(self):
        return self.recording_resolution / 2.0

    @cached_property
    def _zeros_frame_length(self):
        return np.zeros(self.number_of_frames, dtype=np.uint8)

    @cached_property
    def location_sequence_quadrant(self):
        result = self._zeros_frame_length

        result[self.quadrant_upper_left_boolean_index] = 1
        result[self.quadrant_upper_right_boolean_index] = 2
        result[self.quadrant_down_left_boolean_index] = 3
        result[self.quadrant_down_right_boolean_index] = 4

        return reduce_repeating_sequences(result, round(self.fps * 0.35))

    # Quadrant upper left 1

    @cached_property
    def quadrant_upper_left_boolean_index(self):
        return points_in_parallelogram(
            np.array((0.0, 0.0)),
            np.array((self.recording_center_pixel[0], 0.0)),
            np.array((0.0, self.recording_center_pixel[1])),
            self.coordinates_per_frame,
        )

    @cached_property
    def quadrant_upper_left_entries(self):
        return np.sum(self.location_sequence_quadrant == 1)

    @cached_property
    def seconds_on_quadrant_upper_left(self):
        return np.sum(self.quadrant_upper_left_boolean_index) / self.fps

    @cached_property
    def quadrant_upper_left_freezing_time(self):
        return (
            np.sum(
                self.frozen_boolean_index & self.quadrant_upper_left_boolean_index[1:]
            )
            / self.fps
        )

    # Quadrant upper right 2

    @cached_property
    def quadrant_upper_right_boolean_index(self):
        return points_in_parallelogram(
            np.array((self.recording_center_pixel[0], 0.0)),
            self.recording_center_pixel,
            np.array((self.horizontal_resolution, 0.0)),
            self.coordinates_per_frame,
        )

    @cached_property
    def quadrant_upper_right_entries(self):
        return np.sum(self.location_sequence_quadrant == 2)

    @cached_property
    def seconds_on_quadrant_upper_right(self):
        return np.sum(self.quadrant_upper_right_boolean_index) / self.fps

    @cached_property
    def quadrant_upper_right_freezing_time(self):
        return (
            np.sum(
                self.frozen_boolean_index & self.quadrant_upper_right_boolean_index[1:]
            )
            / self.fps
        )

    # Quadrant down left 3

    @cached_property
    def quadrant_down_left_boolean_index(self):
        return points_in_parallelogram(
            np.array((0.0, self.vertical_resolution)),
            np.array((self.recording_center_pixel[0], self.vertical_resolution)),
            np.array((0.0, self.recording_center_pixel[1])),
            self.coordinates_per_frame,
        )

    @cached_property
    def quadrant_down_left_entries(self):
        return np.sum(self.location_sequence_quadrant == 3)

    @cached_property
    def seconds_on_quadrant_down_left(self):
        return np.sum(self.quadrant_down_left_boolean_index) / self.fps

    @cached_property
    def quadrant_down_left_freezing_time(self):
        return (
            np.sum(
                self.frozen_boolean_index & self.quadrant_down_left_boolean_index[1:]
            )
            / self.fps
        )

    # Quadrant down right 4

    @cached_property
    def quadrant_down_right_boolean_index(self):
        return points_in_parallelogram(
            np.array((self.recording_center_pixel[0], self.vertical_resolution)),
            self.recording_resolution,
            self.recording_center_pixel,
            self.coordinates_per_frame,
        )

    @cached_property
    def quadrant_down_right_entries(self):
        return np.sum(self.location_sequence_quadrant == 4)

    @cached_property
    def seconds_on_quadrant_down_right(self):
        return np.sum(self.quadrant_down_right_boolean_index) / self.fps

    @cached_property
    def quadrant_down_right_freezing_time(self):
        return (
            np.sum(
                self.frozen_boolean_index & self.quadrant_down_right_boolean_index[1:]
            )
            / self.fps
        )
