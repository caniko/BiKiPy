from typing import Union, AnyStr, Any, SupportsFloat, Sequence, Dict

import numpy as np

from bikipy.utils.video import seconds_from_frames
from bikipy.feature import movement
from bikipy.reader.deeplabcut import DeepLabCutReader


class BaseExperiment:
    def __init__(
        self,
        coordinate_sequences: Any,
        fps: SupportsFloat,
        unit_per_pixel: SupportsFloat,
        movement_feature_point_label: Union[AnyStr, None] = None,
        label: Any = None,
    ):
        """
        Parameters
        ----------
        coordinate_sequences: Sequence
            The coordinates of the subject across the frames of the video recording
        fps: SupportsFloat
            Number of frames per second
        unit_per_pixel: SupportsFloat
            Number defining the number of pixels that goes into one centimeter
        label: Any; optional
        """

        if movement_feature_point_label:
            # coordinate_sequences must be a reader object, like DeepLabCutReader
            self.movement_feature_point_label = str(movement_feature_point_label)
            self.coordinate_sequences = coordinate_sequences
            self.movement_feature_coordinates = self.coordinate_sequences[
                self.movement_feature_point_label
            ]
        else:
            self.movement_feature_point_label = None
            self.coordinate_sequences = np.asanyarray(coordinate_sequences)
            self.movement_feature_coordinates = self.coordinate_sequences

        self.fps = float(fps)
        self.unit_per_pixel = float(unit_per_pixel)
        self.label = label

        self.displacement, self.mean_speed, self.mean_acceleration = (
            movement.displacement_mean_speed_acceleration(
                self.movement_feature_coordinates,
                self.fps,
                self.unit_per_pixel
            )
        )

    def compute_movement_features_over_boolean_index(
        self, boolean_index: Sequence[bool]
    ):
        boolean_index = np.asanyarray(boolean_index)

        start = None
        displacements, speeds, accelerations = [], [], []
        for i, b_idx in enumerate(boolean_index):
            if b_idx and start is None:
                start = i
            elif not b_idx and start is not None:
                if i - start > self.fps / 3:
                    group_idx = (start, i)

                    displacements.append(
                        (
                            displacement := movement.displacement(
                                self.movement_feature_coordinates[
                                    group_idx[0] : group_idx[1]
                                ],
                            )
                        )
                    )

                    speeds.append((
                        speed := np.abs(np.diff(displacement, axis=0))
                    ))
                    accelerations.append(np.abs(np.diff(speed, axis=0)))

                start = None

        if not displacements:
            return 0, 0, 0

        displacements = np.concatenate(displacements)
        speeds = np.concatenate(speeds)
        accelerations = np.concatenate(accelerations)

        total_seconds = seconds_from_frames(self.fps, displacements.shape[0])

        return (
            np.sum(displacements) * self.unit_per_pixel,
            np.sum(speeds) * self.unit_per_pixel / total_seconds,
            np.sum(accelerations) * self.unit_per_pixel / total_seconds,
        )


class BaseTrial:
    def __init__(
        self,
        exp_id_vs_coordinate_data_path: Dict,
        fps: Union[SupportsFloat, None] = None,
        coordinate_data_format: AnyStr = "deeplabcut",
        **kwargs,
    ):

        self.exp_id_vs_coordinate_data_path = exp_id_vs_coordinate_data_path
        self.fps = float(fps) if fps else None

        self.coordinate_data_format = str(coordinate_data_format).lower()

        if self.coordinate_data_format == "deeplabcut":
            self.exp_id_vs_coordinate_sequences = {
                exp_id: dlc_obj
                for exp_id, dlc_obj in zip(
                    exp_id_vs_coordinate_data_path.keys(),
                    DeepLabCutReader.init_many(
                        exp_id_vs_coordinate_data_path.values(),
                        labels=exp_id_vs_coordinate_data_path.keys(),
                        **kwargs,
                    ),
                )
            }
        else:
            msg = f"{self.coordinate_data_format} as a format for data ingestion has no implementation"
            raise NotImplemented(msg)
