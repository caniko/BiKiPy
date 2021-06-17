from functools import cached_property
from logging import getLogger
from typing import Union, Any

import numpy as np
from matplotlib import pyplot as plt

from bikipy.behaviour.base import BaseTrial
from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.feature.motion import Motion
from bikipy.math.point_in_polygon import points_in_parallelogram


logger = getLogger(__name__)


class SquareEnclosedTrial(BaseTrial):
    def __init__(
        self,
        metric_resolution: Union[Union[float, int], list],
        center_metric_length: Union[float, int],
        **base_trial_kwargs,
    ):
        """
        Parameters
        ----------
        metric_resolution: float
            Length of the square box in which the experiment is conducted
        center_metric_length: float
            Length of the square box signifying periphery and inner area of the
            square box
        base_trial_kwargs
            Keyword arguments passed to BaseTrial
        """
        self.metric_resolution = float(metric_resolution)
        super().__init__(
            unit_per_pixel=(
                self.metric_resolution
                / np.mean(base_trial_kwargs["recording_resolution"])
            ),
            **base_trial_kwargs,
        )

        self.center_metric_length = float(center_metric_length)

        assert self.metric_resolution > self.center_metric_length

        self.center_boolean_indices = points_in_parallelogram(
            self.center_square_corners[0],
            self.center_square_corners[3],
            self.center_square_corners[1],
            self.coordinates_per_frame,
            inspect_points=self.func_inspect,
        )
        self.periphery_boolean_indices = ~self.center_boolean_indices

        self.seconds_on_center = np.sum(self.center_boolean_indices) / self.fps
        self.seconds_on_periphery = np.sum(self.periphery_boolean_indices) / self.fps

        self.center_motion = Motion(
            self.coordinates_per_frame[self.center_boolean_indices],
            self.unit_per_pixel,
            self.fps,
        )
        self.periphery_motion = Motion(
            self.coordinates_per_frame[self.periphery_boolean_indices],
            self.unit_per_pixel,
            self.fps,
        )

        # 1 is center, 2 is periphery, 0 is invalid aka unknown
        self.location_sequence = np.zeros_like(
            self.center_boolean_indices, dtype=np.uint8
        )
        self.location_sequence[self.center_boolean_indices] = 1
        self.location_sequence[self.periphery_boolean_indices] = 2
        self.location_sequence = np.array(
            reduce_repeating_sequences(
                self.location_sequence, frame_tolerance=self._frame_tolerance
            )
        )

        self.center_entries = np.sum(self.location_sequence == 1)
        self.periphery_entries = np.sum(self.location_sequence == 2)

    @cached_property
    def center_square_corners(self):
        def non_square_rectification(x_bias: float = 0.0, y_bias: float = 0.0):
            if (x_bias := float(x_bias)) and (y_bias := float(y_bias)):
                raise ValueError

            if x_bias:
                y_short = self.vertical_resolution * center_box_ratio
                y_long = self.vertical_resolution * center_box_ratio_minus_1

                x_short = y_short + x_bias
                x_long = y_long + x_bias

            else:
                x_short = self.horizontal_resolution * center_box_ratio
                x_long = self.horizontal_resolution * center_box_ratio_minus_1

                y_short = x_short + y_bias
                y_long = x_long + y_bias

            return (
                (x_short, y_short),
                (x_short, y_long),
                (x_long, y_long),
                (x_long, y_short),
            )

        center_box_ratio = (
            (self.metric_resolution - self.center_metric_length) / 2.0
        ) / self.metric_resolution
        center_box_ratio_minus_1 = center_box_ratio - 1

        x = self.horizontal_resolution * center_box_ratio
        x_rest_half = (self.horizontal_resolution - x) / 2.0

        y = self.vertical_resolution * center_box_ratio
        y_rest_half = (self.vertical_resolution - y) / 2.0
        if self.horizontal_resolution == self.vertical_resolution:
            return (
                # x_short, y_short
                (x_rest_half, y_rest_half),
                # x_short, y_long
                (x_rest_half, self.vertical_resolution - y_rest_half),
                # x_long, y_long
                (
                    self.horizontal_resolution - x_rest_half,
                    self.vertical_resolution - y_rest_half,
                ),
                # x_long, y_short
                (self.horizontal_resolution - x_rest_half, y_rest_half),
            )
        elif self.horizontal_resolution < self.vertical_resolution:
            return non_square_rectification(
                y_bias=(self.vertical_resolution - self.horizontal_resolution) / 2.0
            )
        else:
            return non_square_rectification(
                x_bias=(self.horizontal_resolution - self.vertical_resolution) / 2.0
            )

    @cached_property
    def center_freezing_time(self):
        return (
            np.sum(self.frozen_boolean_indices & self.center_boolean_indices[1:])
            / self.fps
        )

    @cached_property
    def periphery_freezing_time(self):
        return (
            np.sum(self.frozen_boolean_indices & self.periphery_boolean_indices[1:])
            / self.fps
        )

    def plot(self, ax: Any = None):
        if not ax:
            fig, ax = plt.subplots()
        if self.inspect_image is not None:
            ax.imshow(self.inspect_image)
        else:
            logger.warning("inspect_image is not defined will plot without")

        for i in range((length := len(self.center_square_corners))):
            next_i = i + 1
            ax.plot(
                self.center_square_corners[i],
                self.center_square_corners[next_i if next_i != length else 0],
            )

        return ax

    @property
    def motion_info(self):
        return (
            self.motion.to_list()
            + self.periphery_motion.to_list()
            + self.center_motion.to_list()
        )

    @property
    def feature_info(self):
        return [
            self.total_freezing_time,
            self.center_freezing_time,
            self.periphery_freezing_time,
            self.seconds_on_periphery,
            self.seconds_on_center,
            self.periphery_entries,
            self.center_entries,
        ]

    @property
    def info(self):
        return self.motion_info + self.feature_info
