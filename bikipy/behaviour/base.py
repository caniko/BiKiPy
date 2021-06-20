from functools import cached_property
from logging import getLogger
from typing import Any, Sequence, Union

import numpy as np

from bikipy.feature.motion import Motion, freezing_time, displacement_per_frame
from bikipy.reader.deeplabcut import DeepLabCutReader

logger = getLogger(__name__)


class BaseTrial:
    second_tolerance = 0.35

    def __init__(
        self,
        coordinate_sequence: Any,
        unit_per_pixel: float,
        rigid_nodes_freezing: Union[Sequence[Union[str, int]], None] = None,
        recording_resolution: Union[Sequence[int], None] = None,
        movement_feature_point_label: Union[str, None] = None,
        fps: Union[float, None] = None,
        label: Any = None,
        func_inspect: bool = False,
        inspect_image: Any = None,
    ):
        """
        Parameters
        ----------
        coordinate_sequence: Sequence
            The coordinates of the subject across the frames in the video recording
        fps: float
            Number of frames per second
        unit_per_pixel: float
            Number defining the number of pixels that goes into one centimeter
        label: Any; optional
        """

        self.fps = fps
        self.unit_per_pixel = float(unit_per_pixel)
        self.label = label
        self.func_inspect = func_inspect
        self.inspect_image = inspect_image

        if recording_resolution:
            assert len(recording_resolution) == 2, recording_resolution
            self.horizontal_resolution = int(recording_resolution[0])
            self.vertical_resolution = int(recording_resolution[1])
            self.recording_resolution = (
                self.horizontal_resolution,
                self.vertical_resolution,
            )

        # coordinate_sequence must be a reader object, like DeepLabCutReader
        self.movement_feature_point_label = str(movement_feature_point_label)
        self.coordinate_sequence: dict = coordinate_sequence
        self.coordinates_per_frame = self.coordinate_sequence[
            self.movement_feature_point_label
        ]

        self.experiment_seconds = self.coordinates_per_frame.shape[0] / self.fps

        self.motion = Motion(self.coordinates_per_frame, self.unit_per_pixel, self.fps)

        self._rigid_nodes_freezing = None
        self._frozen_boolean_indices = None
        if rigid_nodes_freezing:
            self.rigid_nodes_freezing = rigid_nodes_freezing

    @cached_property
    def _frame_tolerance(self):
        return round(self.second_tolerance * self.fps)

    @property
    def rigid_nodes_freezing(self):
        return self._rigid_nodes_freezing

    @rigid_nodes_freezing.setter
    def rigid_nodes_freezing(self, value: Sequence):
        self._rigid_nodes_freezing = value
        if not value:
            return

        self._frozen_boolean_indices = freezing_time(
            self.fps,
            [
                displacement_per_frame(coordinate_sequence, remove_tails=False)
                * self.unit_per_pixel
                for coordinate_sequence in self.coordinate_sequence[
                    self.rigid_nodes_freezing
                ]
            ],
        )

    @property
    def frozen_boolean_indices(self):
        if self._frozen_boolean_indices is None:
            msg = "rigid_nodes_freezing has to be defined in order to compute frozen time data"
            raise AttributeError(msg)
        return self._frozen_boolean_indices

    @cached_property
    def total_freezing_time(self):
        return np.sum(self.frozen_boolean_indices) / self.fps


class BaseExperiment:
    def __init__(
        self,
        trial_id_vs_coordinate_data_path: dict,
        fps: Union[dict, float, None] = None,
        coordinate_data_format: str = "deeplabcut",
        label: Any = None,
        func_inspect: bool = False,
        **data_import_kwargs,
    ):
        if fps and not isinstance(fps, (float, int, dict)):
            msg = f"fps has to be float, int or dict, and not {type(fps)}"
            raise ValueError(msg)

        self.trial_id_vs_coordinate_data_path = trial_id_vs_coordinate_data_path
        self.fps = fps

        self.coordinate_data_format = str(coordinate_data_format).lower()
        self.label, self.func_inspect = label, func_inspect

        if self.coordinate_data_format == "deeplabcut":
            self.exp_id_vs_coordinate_sequences = {
                exp_id: dlc_obj
                for exp_id, dlc_obj in zip(
                    trial_id_vs_coordinate_data_path.keys(),
                    DeepLabCutReader.init_many(
                        trial_id_vs_coordinate_data_path.values(),
                        labels=trial_id_vs_coordinate_data_path.keys(),
                        **data_import_kwargs,
                    ),
                )
            }
        else:
            msg = f"{self.coordinate_data_format} as a format for data ingestion has no implementation"
            raise NotImplemented(msg)
