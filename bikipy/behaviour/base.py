from functools import lru_cache
from logging import getLogger
from typing import Any, Iterable, Sequence, Union

import numpy as np
import pandas as pd

from bikipy.feature.motion import Motion, total_displacement_median_speed_acceleration
from bikipy.reader.deeplabcut import DeepLabCutReader
from bikipy.utils.misc import resolve_stem_in_filepath

logger = getLogger(__name__)


class BaseTrial:
    def __init__(
        self,
        coordinate_sequence: Any,
        unit_per_pixel: float,
        fps: Union[float, None] = None,
        recording_resolution: Union[Iterable[int], None] = None,
        movement_feature_point_label: Union[str, None] = None,
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

        if movement_feature_point_label:
            # coordinate_sequence must be a reader object, like DeepLabCutReader
            self.movement_feature_point_label = str(movement_feature_point_label)
            self.coordinate_sequence: dict = coordinate_sequence
            self.coordinates_per_frame = self.coordinate_sequence[
                self.movement_feature_point_label
            ]
        else:
            self.movement_feature_point_label = None
            self.coordinate_sequence: np.ndarray = np.asarray(coordinate_sequence)
            self.coordinates_per_frame = self.coordinate_sequence

        self.experiment_seconds = self.coordinates_per_frame.shape[0] / self.fps

        self.motion = Motion(self.coordinates_per_frame, self.unit_per_pixel, self.fps)


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
