import os
from functools import cached_property, lru_cache
from logging import getLogger
from pathlib import Path, PurePath
from typing import (
    Any,
    Iterable,
    Literal,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Union,
)

import cv2
import numpy as np
import pandas as pd
from pydantic import DirectoryPath, Field, FilePath
from tqdm import tqdm

from bikipy._base_class import BikipyBase
from bikipy.feature.motion import Motion, displacement_by_frame, frozen_frames
from bikipy.reader.base import BaseReader
from bikipy.reader.deeplabcut import DeepLabCutReader
from bikipy.utils.store import RangeDict
from bikipy.utils.typing import NDArray
from bikipy.utils.video import get_video_data

logger = getLogger(__name__)


class Behaviour(BikipyBase):
    fps: Union[float, int, None] = None
    recording_resolution: Optional[NDArray[Literal["np.int16"]]] = None
    metric_resolution: Union[np.ndarray, float, None] = None
    manual_units_per_pixel: Optional[float] = None
    _live: bool = Field(False)

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


class BaseExperiment(Behaviour):
    trial_id_vs_data: Optional[dict] = None
    trial_id_range_vs_common_data: Union[RangeDict, dict, None] = None
    point_label_for_motion_features: Optional[str] = None
    inspection_figure_save: Union[bool, DirectoryPath] = Field(False)
    data_import_kwargs: Optional[dict] = None
    coordinate_data_format: Literal["deeplabcut"] = Field("deeplabcut")

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

    @cached_property
    def trial_id_vs_coordinate_sequence(self):
        if self.coordinate_data_format == "deeplabcut":
            return {
                trial_id: dlc_obj
                for trial_id, dlc_obj in zip(
                    self._trial_id_iterable,
                    DeepLabCutReader.init_many_map(
                        data_path=(
                            data["coordinate_data_path"]
                            for data in self.trial_id_vs_data.values()
                        ),
                        labels=self.trial_id_vs_data.keys(),
                        **self.data_import_kwargs,
                    ),
                )
            }
        msg = f"{self.coordinate_data_format} as a format for data ingestion has no implementation"
        raise NotImplemented(msg)

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

    @cached_property
    def length(self):
        return len(self._trial_id_iterable)

    @property
    def number_of_trials(self):
        return self.length

    @property
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
        return pd.Series(self._trial_id_iterable, name="Test ID", dtype=np.int16)

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
    def instanced_trial_data(self):
        raise NotImplementedError

    @property
    def summary_frame(self):
        return pd.DataFrame(
            (trial_data.motion.to_list for trial_data in self.instanced_trial_data),
            columns=self._motion_2d_multi_indexer("All"),
            index=self.frame_index,
        )


class BaseTrial(Behaviour):
    coordinate_sequence: Optional[BaseReader] = Field(
        description="The coordinates of the subject across the frames in the video recording"
    )
    animal_id: Optional[int] = Field(description="The ID of the animal in the trial")
    rigid_nodes_freezing: Optional[Sequence[Union[str, int]]] = Field(
        description="Nodes that should remain during freeze/immobility, most often due to fear."
    )
    point_label_for_motion_features: Optional[str] = Field(
        description="Label of the node that will be used to track general animal movement"
    )
    video_path: Optional[FilePath] = Field(description="Path to trial video recording")
    inspection_figure_save: Union[PurePath, str, bool] = Field(
        description="Path to save figures for inspection of results"
    )
    inspect_image: Optional[FilePath] = Field(description="Image used for inspection")

    _category = "trial"

    _trial_sequence_index = None
    _trial_label = None
    _second_tolerance = 0.35

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

        # Variables for trials with zones, see doc for more info.
        self.perimeters = None
        self.trial_start_perimeter = None
        # ========================= ========================= =========================
        self._rigid_nodes_freezing = None
        self._frozen_boolean_index = None

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
