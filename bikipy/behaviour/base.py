import datetime
import os
from functools import cached_property
from logging import getLogger
from pathlib import Path, PurePath
from typing import Any, Iterable, Sequence, Union

import compress_pickle
import cv2
import numpy as np
from tqdm import tqdm

from bikipy.feature.motion import Motion, displacement_by_frame, frozen_frames
from bikipy.reader.deeplabcut import DeepLabCutReader
from bikipy.utils.store import RangeDict
from bikipy.utils.video import get_video_data

logger = getLogger(__name__)


class Behaviour:
    category = None

    def __init__(
        self,
        label: Any = None,
        timestamp: Any = None,
        save_root: Union[PurePath, str, None] = None,
        _live: bool = False,
    ):
        self.label = label
        self.timestamp = timestamp or datetime.datetime.now()
        self.save_root = Path(save_root) if save_root else None

        self._live = _live

    def save(self, save_root: Union[PurePath, str, None] = None):
        save_root = Path(save_root or self.save_root)
        assert save_root
        compress_pickle.dump(
            self, save_root / f"pickle_{self.category}_{self.timestamp}.lzma"
        )

    @property
    def __hash_key(self):
        raise NotImplemented

    def __hash__(self):
        return sum(hash(key) for key in self.__hash_key)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__hash_key == other.__hash_key
        return self.__hash_key == other

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def _assert_timestamp_attribute(obj):
        assert hasattr(obj, "timestamp")

    def __lt__(self, other):
        self._assert_timestamp_attribute(other)
        return self.timestamp < other.timestamp

    def __le__(self, other):
        self._assert_timestamp_attribute(other)
        return self.timestamp <= other.timestamp

    def __gt__(self, other):
        self._assert_timestamp_attribute(other)
        return self.timestamp > other.timestamp

    def __ge__(self, other):
        self._assert_timestamp_attribute(other)
        return self.timestamp >= other.timestamp


class BaseExperiment(Behaviour):
    category = "experiment"
    trials_are_sequential = False

    def __init__(
        self,
        metric_resolution: Union[float, Sequence[float]],
        trial_id_vs_data: dict,
        trial_id_range_vs_data: Union[dict, None] = None,
        coordinate_data_format: str = "deeplabcut",
        inspection_figure_save: Union[bool, str, PurePath] = False,
        data_import_kwargs: Union[dict, None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.trial_id_vs_data = dict(trial_id_vs_data)
        self._trial_ids_iterable = self.trial_id_vs_data.keys()
        self.trial_ids = tuple(self._trial_ids_iterable)

        self.trial_id_range_vs_data = (
            RangeDict(trial_id_range_vs_data) if trial_id_range_vs_data else None
        )
        self.metric_resolution = metric_resolution

        self.coordinate_data_format = str(coordinate_data_format).lower()
        if isinstance(inspection_figure_save, bool):
            self.inspection_figure_save = inspection_figure_save
        elif isinstance(inspection_figure_save, str) or isinstance(
            inspection_figure_save, PurePath
        ):
            self.inspection_figure_save = (
                inspection_figure_save / f"Experiment_{self.label}_inspect"
            )
            os.mkdir(self.inspection_figure_save)

        if self.coordinate_data_format == "deeplabcut":
            self.trial_id_vs_coordinate_sequences = {
                exp_id: dlc_obj
                for exp_id, dlc_obj in zip(
                    self._trial_ids_iterable,
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
    def __hash_key(self):
        return self.trial_id_vs_data

    def __getitem__(self, item):
        if self.trial_id_range_vs_data:
            return {**self.trial_id_vs_data[item], **self.trial_id_range_vs_data[item]}
        else:
            return self.trial_id_vs_data[item]

    @cached_property
    def length(self):
        return len(self.trial_ids)

    def trial_id_data_tqdm(self):
        return tqdm(
            ((trial_id, self[trial_id]) for trial_id in self.trial_ids),
            total=self.length,
        )


class BaseTrial(Behaviour):
    category = "trial"

    trial_sequence_index = None
    second_tolerance = 0.35

    def __init__(
        self,
        coordinate_sequence: Union[dict, None] = None,
        animal_id: Union[int, None] = None,
        metric_resolution: Union[Union[float, int], list, None] = None,
        rigid_nodes_freezing: Union[Sequence[Union[str, int]], None] = None,
        movement_feature_point_label: Union[str, None] = None,
        video_path: Any = None,
        recording_resolution: Union[Sequence[int], None] = None,
        fps: Union[float, int, None] = None,
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
        :param movement_feature_point_label: Label of the node that will be used to track general animal movement
        :param recording_resolution: Video resolution
        :param label: Experiment label
        :param inspection_figure_save: Path to save figures for inspection of results
        :param inspect_image: Image used for inspection

        :type coordinate_sequence: dict
        :type video_path: Any
        :type metric_resolution: int or float, or [(float, int), (float, int)]
        :type animal_id: int (optional)
        :type rigid_nodes_freezing: Sequence[Union[str, int]] (optional)
        :type movement_feature_point_label: str (optional)
        :type recording_resolution: Sequence[int] (optional)
        :type label: Any (optional)
        :type inspection_figure_save: bool
        :type inspect_image: Any
        """

        super().__init__(**kwargs)

        self.inspect_image = inspect_image
        self.video_path = video_path

        if self.video_path:
            (
                _frame,
                self.horizontal_resolution,
                self.vertical_resolution,
                self.fps,
            ) = get_video_data(video_path)
            self.recording_resolution = np.array(
                (
                    self.horizontal_resolution,
                    self.vertical_resolution,
                )
            )
        elif recording_resolution and fps:
            self.fps = fps
            self.recording_resolution = recording_resolution
            self.horizontal_resolution, self.vertical_resolution = recording_resolution
        elif self.inspect_image and fps:
            self.fps = fps
            self.vertical_resolution, self.horizontal_resolution = cv2.imread(
                self.inspect_image
            ).shape[:-1]
        else:
            msg = (
                "recording_resolution and fps could not be defined."
                "One of the following compbinations must be provided:\n"
                "\t1. Trial video path\n"
                "\t2. recording_resolution and frame per second (fps)\n"
                "\t3. inspect_image and frame per second (fps)"
            )
            raise ValueError(msg)

        self.animal_id = int(animal_id) if animal_id else None

        self.metric_resolution = metric_resolution if metric_resolution else None
        if self.metric_resolution:
            if isinstance(self.metric_resolution, (float, int)):
                self.unit_per_pixel = self.metric_resolution / np.mean(
                    self.recording_resolution
                )
            else:
                self.unit_per_pixel = (
                    np.array(self.metric_resolution) / self.recording_resolution
                )

        self.inspection_figure_save = inspection_figure_save

        if not self._live:
            # coordinate_sequence must be a reader object, like DeepLabCutReader
            self.movement_feature_point_label = str(movement_feature_point_label)
            self.coordinate_sequence = coordinate_sequence
            self.coordinates_per_frame = self.coordinate_sequence[
                self.movement_feature_point_label
            ]

            self.experiment_seconds = self.coordinates_per_frame.shape[0] / self.fps

        # Variables for trials with zones, see doc for more info.
        self.perimeters = None
        self.trial_start_perimeter = None
        # ========================= ========================= =========================

        self._rigid_nodes_freezing = None
        self._frozen_boolean_index = None
        if rigid_nodes_freezing:
            self.rigid_nodes_freezing = rigid_nodes_freezing

    @cached_property
    def motion(self):
        return Motion(self.coordinates_per_frame, self.unit_per_pixel, self.fps)

    @property
    def __hash_key(self):
        return self.coordinates_per_frame, self.fps, self.metric_resolution

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
                * self.unit_per_pixel
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
