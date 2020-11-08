from typing import Union, AnyStr, Any, SupportsFloat, Sequence, Dict

import numpy as np

from bikipy.features.movement import displacement_mean_speed_acceleration
from bikipy.readers.deeplabcut import DeepLabCutReader


class BaseExperiment:
    def __init__(
        self,
        coordinate_sequence: Any,
        fps: SupportsFloat,
        cm_per_pixel: SupportsFloat,
        movement_feature_point_label: Union[AnyStr, None] = None,
        label: Any = None,
    ):
        """
        Parameters
        ----------
        coordinate_sequence: Sequence
            The coordinates of the subject across the frames of the video recording
        fps: SupportsFloat
            Number of frames per second
        cm_per_pixel: SupportsFloat
            Number defining the number of pixels that goes into one centimeter
        label: Any; optional
        """
        self.coordinate_sequence = np.asanyarray(coordinate_sequence)
        if movement_feature_point_label:
            self.movement_feature_point_label = str(movement_feature_point_label)

        self.fps = float(fps)
        self.cm_per_pixel = float(cm_per_pixel)
        self.label = label

        self.displacement, self.mean_speed, self.mean_acceleration = (
            displacement_mean_speed_acceleration(
                self.movement_feature_coordinates,
                self.fps,
            )
            * self.cm_per_pixel
        )

    @property
    def movement_feature_coordinates(self):
        return np.asanyarray(
            self.coordinate_sequence
            if self.movement_feature_point_label
            else self.coordinate_sequence[self.movement_feature_point_label]
        )

    def compute_movement_features_over_boolean_index(
        self, boolean_index: Sequence[bool]
    ):
        boolean_index = np.asanyarray(boolean_index)

        start = None
        displacements, mean_speeds, mean_accelerations = [], [], []
        for i, b_idx in enumerate(boolean_index):
            if b_idx and start is None:
                start = i
            elif not b_idx and start is not None:
                group_idx = (start, i)
                (
                    displacement,
                    mean_speed,
                    mean_acceleration,
                ) = displacement_mean_speed_acceleration(
                    self.movement_feature_coordinates[group_idx[0] : group_idx[1]],
                    self.fps,
                )

                displacements.append(displacements)
                mean_speeds.append(mean_speed)
                mean_accelerations.append(mean_acceleration)

                start = None

        return (
            np.array(
                np.sum(displacements), np.mean(mean_speeds), np.mean(mean_accelerations)
            )
            * self.cm_per_pixel
        )


class BaseTrial:
    def __init__(self, exp_id_vs_coordinate_data_path: Dict, coordinate_data_format: AnyStr = "deeplabcut"):
        self.coordinate_data_format = str(coordinate_data_format)

        if self.coordinate_data_format == "deeplabcut":
            self.exp_id_vs_coordinate_sequences = {
                exp_id: dlc_obj for exp_id, dlc_obj in zip(
                    exp_id_vs_coordinate_data_path.keys(),
                    DeepLabCutReader.init_many()
                )
            }
