import os
import datetime
from functools import cached_property, lru_cache
from logging import getLogger
from pathlib import Path, PurePath
from typing import Any, Iterable, Sequence, Union

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from bikipy._base_class import BikipyBase
from bikipy.feature.motion import Motion, displacement_by_frame, frozen_frames
from bikipy.reader.base import BaseReader
from bikipy.reader.deeplabcut import DeepLabCutReader
from bikipy.utils.store import RangeDict
from bikipy.utils.video import get_video_data

logger = getLogger(__name__)


class Behaviour(BikipyBase):
    def __init__(
        self,
        *args,
        fps: Union[float, int, None] = None,
        recording_resolution: Union[Sequence[int], None] = None,
        metric_resolution: Union[Sequence[float], None] = None,
        units_per_pixel: Union[float, None] = None,
        _live: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.fps = fps

        self.recording_resolution = recording_resolution

        self.metric_resolution = metric_resolution
        self.units_per_pixel = units_per_pixel

        self._live = _live

    @property
    def recording_resolution(self):
        return self._recording_resolution

    @recording_resolution.setter
    def recording_resolution(self, value: Sequence):
        self._recording_resolution = (
            np.asarray(value, dtype=np.int16) if np.any(value) else None
        )

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
        return self._user_defined_units_per_pixel or self.computed_units_per_pixel

    @units_per_pixel.setter
    def units_per_pixel(self, value: Union[float, None]):
        self._user_defined_units_per_pixel = value

    @cached_property
    def computed_units_per_pixel(self):
        if not np.any(self.metric_resolution):
            msg = "metric_resolution attribute needs to be defined to compute units_per_pixel"
            raise AttributeError(msg)
        if isinstance(self.metric_resolution, (float, int)):
            return self.metric_resolution / np.mean(self.recording_resolution)
        else:
            return np.array(self.metric_resolution) / self.recording_resolution


class BaseExperiment(Behaviour):
    category = "experiment"
    trials_are_sequential = False

    def __init__(
        self,
        trial_id_vs_data: Union[dict, None] = None,
        trial_id_range_vs_common_data: Union[dict, None] = None,
        coordinate_data_format: str = "deeplabcut",
        point_label_for_motion_features: Union[str, None] = None,
        inspection_figure_save: Union[bool, str, PurePath] = False,
        data_import_kwargs: Union[dict, None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.trial_id_vs_data = dict(trial_id_vs_data)
        self.trial_id_range_vs_common_data = (
            RangeDict(trial_id_range_vs_common_data)
            if trial_id_range_vs_common_data
            else None
        )

        self.point_label_for_motion_features = (
            str(point_label_for_motion_features)
            if point_label_for_motion_features
            else None
        )
        self.coordinate_data_format = str(coordinate_data_format).lower()
        if isinstance(inspection_figure_save, bool):
            self.inspection_figure_save = inspection_figure_save
        elif isinstance(inspection_figure_save, (PurePath, str)):
            self.inspection_figure_save = (
                inspection_figure_save / f"Experiment_{self.timestamp}_inspect"
            )
            if not inspection_figure_save.exists():
                os.makedirs(self.inspection_figure_save)

        if self.coordinate_data_format == "deeplabcut":
            self.trial_id_vs_coordinate_sequence = {
                trial_id: dlc_obj
                for trial_id, dlc_obj in zip(
                    self._trial_id_iterable,
                    DeepLabCutReader.init_many(
                        (
                            data["coordinate_data_path"]
                            for data in self.trial_id_vs_data.values()
                        ),
                        labels=self.trial_id_vs_data.keys(),
                        **data_import_kwargs,
                    ),
                )
            }
        else:
            msg = f"{self.coordinate_data_format} as a format for data ingestion has no implementation"
            raise NotImplemented(msg)

    @property
    def _hash_key(self):
        return self.trial_id_vs_data

    def __getitem__(self, item):
        if not self.trial_id_range_vs_common_data:
            return self.trial_id_vs_data[item]
        return {
            **self.trial_id_range_vs_common_data[item],
            **self.trial_id_vs_data[item],
        }

    @lru_cache
    def generic_trial_kwargs(self, trial_id: int):
        trial_meta = self[trial_id]
        generic_kwargs = {
            "int_label": trial_id,
            "coordinate_sequence": self.trial_id_vs_coordinate_sequence[trial_id],
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
    def length(self):
        return len(self._trial_id_iterable)

    @property
    def number_of_trials(self):
        return self.length

    def trial_id_data_tqdm(self):
        return tqdm(
            ((trial_id, self[trial_id]) for trial_id in self._trial_id_iterable),
            total=self.length,
        )

    @property
    def _trial_id_iterable(self):
        return self.trial_id_vs_data.keys()

    @cached_property
    def frame_index(self):
        return pd.Series(self._trial_id_iterable, name="Test", dtype=np.int16)

    @staticmethod
    def _motion_2d_multi_indexer(category: str):
        category = str(category)
        return (
            (category, "Displacement"),
            (category, "Median_speed"),
            (category, "Median_acceleration"),
            (category, "Freezing time"),
        )

    @staticmethod
    def _feature_2d_multi_indexer(feature: str, category):
        return tuple([(feature, category) for category in category])


class BaseTrial(Behaviour):
    category = "trial"

    trial_sequence_index = None
    trial_label = None
    second_tolerance = 0.35

    def __init__(
        self,
        coordinate_sequence: Union[BaseReader, None] = None,
        animal_id: Union[int, None] = None,
        rigid_nodes_freezing: Union[Sequence[Union[str, int]], None] = None,
        point_label_for_motion_features: Union[str, None] = None,
        video_path: Any = None,
        inspection_figure_save: Union[PurePath, str, bool] = False,
        inspect_image: Any = None,
        **kwargs,
    ):
        """
        :param coordinate_sequence: The coordinates of the subject across the frames in the video recording
        :param video_path: Path to trial video recording
        :param animal_id: The ID of the animal in the trial
        :param metric_resolution: Length of the square box in which the experiment is conducted
        :param rigid_nodes_freezing: Nodes that should remain during freeze/immobility, most often due to fear.
        :param point_label_for_motion_features: Label of the node that will be used to track general animal movement
        :param recording_resolution: Video resolution
        :param label: Experiment label
        :param inspection_figure_save: Path to save figures for inspection of results
        :param inspect_image: Image used for inspection

        :type coordinate_sequence: BaseReader
        :type video_path: Any
        :type metric_resolution: int or float, or [(float, int), (float, int)]
        :type animal_id: int (optional)
        :type rigid_nodes_freezing: Sequence[Union[str, int]] (optional)
        :type point_label_for_motion_features: str (optional)
        :type recording_resolution: Sequence[int] (optional)
        :type label: Any (optional)
        :type inspection_figure_save: bool
        :type inspect_image: Any
        """

        super().__init__(**kwargs)

        if video_path:
            (
                _frame,
                horizontal_resolution,
                vertical_resolution,
                self.fps,
            ) = get_video_data(video_path)
            self.recording_resolution = horizontal_resolution, vertical_resolution
        elif self.recording_resolution and self.fps:
            pass
        elif inspect_image and self.fps:
            self.recording_resolution = cv2.imread(inspect_image).shape[:-1]
        else:
            msg = (
                "recording_resolution and fps could not be defined."
                "One of the following compbinations must be provided:\n"
                "\t1. Trial video path\n"
                "\t2. recording_resolution and frame per second (fps)\n"
                "\t3. inspect_image and frame per second (fps)"
            )
            raise ValueError(msg)

        self.inspect_image = inspect_image
        self.video_path = video_path
        self.animal_id = int(animal_id) if animal_id else None

        self.inspection_figure_save = inspection_figure_save

        if not self._live:
            # coordinate_sequence must be a reader object, like DeepLabCutReader
            self.point_label_for_motion_features = point_label_for_motion_features
            self.coordinate_sequence = coordinate_sequence

        # Variables for trials with zones, see doc for more info.
        self.perimeters = None
        self.trial_start_perimeter = None
        # ========================= ========================= =========================

        self._rigid_nodes_freezing = None
        self._frozen_boolean_index = None
        if rigid_nodes_freezing:
            self.rigid_nodes_freezing = rigid_nodes_freezing

    @property
    def coordinates_per_frame(self):
        return self.coordinate_sequence[self.point_label_for_motion_features]

    @cached_property
    def experiment_seconds(self):
        return self.coordinates_per_frame.shape[0] / self.fps

    @cached_property
    def motion(self):
        return Motion(self.coordinates_per_frame, self.units_per_pixel, self.fps)

    @property
    def _hash_key(self):
        return self.coordinates_per_frame

    @property
    def inspect_image_path(self):
        if isinstance(self.inspection_figure_save, str) or isinstance(
            self.inspection_figure_save, PurePath
        ):
            return Path(self.inspection_figure_save) / self.label
        return self.inspection_figure_save  # return the bool in any case

    @cached_property
    def _frame_tolerance(self):
        return round(self.second_tolerance * self.fps)

    @property
    def rigid_nodes_freezing(self):
        return self._rigid_nodes_freezing

    @rigid_nodes_freezing.setter
    def rigid_nodes_freezing(self, value: Sequence):
        if not value:
            return

        self._rigid_nodes_freezing = value
        self._frozen_boolean_index = frozen_frames(
            self.fps,
            [
                displacement_by_frame(coordinate_sequence, remove_tails=False)
                * self.units_per_pixel
                for coordinate_sequence in self.coordinate_sequence[
                    self.rigid_nodes_freezing
                ]
            ],
        )

    @property
    def frozen_boolean_index(self):
        if self._frozen_boolean_index is None:
            msg = "rigid_nodes_freezing has to be defined in order to compute frozen time data"
            raise AttributeError(msg)
        return self._frozen_boolean_index

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
        return [self.trial_label] if self.trial_label else []
