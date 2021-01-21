from logging import getLogger
from typing import Any, AnyStr, Dict, Sequence, SupportsFloat, SupportsInt, Union

import numpy as np
import pandas as pd

from bikipy.feature import movement
from bikipy.reader.deeplabcut import DeepLabCutReader
from bikipy.utils.misc import resolve_stem_in_filepath

logger = getLogger(__name__)


class BaseExperiment:
    def __init__(
        self,
        coordinate_sequence: Any,
        fps: SupportsFloat,
        length_unit_per_pixel: SupportsFloat,
        recording_resolution: Union[Sequence[SupportsInt], None] = None,
        movement_feature_point_label: Union[AnyStr, None] = None,
        label: Any = None,
        guiding_image: Any = None,
    ):
        """
        Parameters
        ----------
        coordinate_sequence: Sequence
            The coordinates of the subject across the frames in the video recording
        fps: SupportsFloat
            Number of frames per second
        length_unit_per_pixel: SupportsFloat
            Number defining the number of pixels that goes into one centimeter
        label: Any; optional
        """

        self.fps = float(fps)
        self.length_unit_per_pixel = float(length_unit_per_pixel)
        self.label = label
        self.guiding_image = guiding_image

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
            self.coordinate_sequence = coordinate_sequence
            self.coordinates_per_frame = self.coordinate_sequence[
                self.movement_feature_point_label
            ]
        else:
            self.movement_feature_point_label = None
            self.coordinate_sequence = np.asanyarray(coordinate_sequence)
            self.coordinates_per_frame = self.coordinate_sequence

        self.experiment_seconds = self.coordinates_per_frame.shape[0] / self.fps

        self.displacement_per_frame = movement.displacement_per_frame(
            self.coordinates_per_frame
        )
        self.acceleration_per_frame = np.abs(
            np.diff(self.displacement_per_frame, axis=0)
        )

        (
            self.displacement,
            self.mean_speed,
            self.mean_acceleration,
        ) = movement.displacement_mean_speed_acceleration(
            self.coordinates_per_frame, self.fps, self.length_unit_per_pixel
        )

    def compute_movement_features_over_boolean_index(
        self, boolean_index: Sequence[bool]
    ):
        boolean_index = np.asanyarray(boolean_index)

        start = None
        displacements, accelerations = [], []
        for i, b_idx in enumerate(boolean_index):
            if b_idx and start is None:
                start = i
            elif not b_idx and start is not None:
                if i - start <= self.fps / 3:
                    continue

                displacements.append(self.displacement_per_frame[start : i - 1])
                accelerations.append(self.acceleration_per_frame[start : i - 2])

                start = None

        if not displacements:
            return 0, 0, 0

        displacements = np.concatenate(displacements)
        accelerations = np.concatenate(accelerations)

        unit_converter = movement.units_pixels_per_second_frame(
            self.length_unit_per_pixel, self.fps
        )

        return (
            np.sum(displacements) * self.length_unit_per_pixel,  # total_displacement
            np.mean(displacements) * unit_converter,  # average speed
            np.mean(accelerations) * unit_converter,  # average acceleration
        )


class BaseTrial:
    def __init__(
        self,
        exp_id_vs_coordinate_data_path: Dict,
        fps: Union[Dict, SupportsFloat, None] = None,
        coordinate_data_format: AnyStr = "deeplabcut",
        label: Any = None,
        debug: bool = False,
        **init_kwargs,
    ):

        self.exp_id_vs_coordinate_data_path = exp_id_vs_coordinate_data_path
        self.fps = fps

        self.coordinate_data_format = str(coordinate_data_format).lower()
        self.label, self.debug = label, debug

        if self.coordinate_data_format == "deeplabcut":
            self.exp_id_vs_coordinate_sequences = {
                exp_id: dlc_obj
                for exp_id, dlc_obj in zip(
                    exp_id_vs_coordinate_data_path.keys(),
                    DeepLabCutReader.init_many(
                        exp_id_vs_coordinate_data_path.values(),
                        labels=exp_id_vs_coordinate_data_path.keys(),
                        **init_kwargs,
                    ),
                )
            }
        else:
            msg = f"{self.coordinate_data_format} as a format for data ingestion has no implementation"
            raise NotImplemented(msg)

    def to_ods(self, filepath: Any) -> None:
        filepath = resolve_stem_in_filepath(filepath)
        result = dict(self.export_to_dataframe())

        with pd.ExcelWriter(filepath) as writer:
            for time, trial in result.items():
                trial.to_excel(writer, sheet_name=time)

        logger.info(f"ODS file saved to: {filepath}")
