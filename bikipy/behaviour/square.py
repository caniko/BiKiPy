from functools import cached_property
from logging import getLogger
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from bikipy.behaviour.base import BaseExperiment, BaseTrial
from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.math.point_in_polygon import points_in_parallelogram

logger = getLogger(__name__)


class SquareEnclosedExperiment(BaseExperiment):
    global_center_metric_length: Optional[float]

    def trial_keyword_arguments(self, trial_id: int) -> dict:
        generic = super().trial_keyword_arguments(trial_id)

        if self.global_center_metric_length:
            generic["center_metric_length"] = self.global_center_metric_length

        return generic

    @cached_property
    def motion_summary_frame(self):
        periphery_center = ("Periphery", "Center")
        return pd.concat(
            (
                super().motion_summary_frame,
                pd.DataFrame(
                    (
                        trial.center_motion.to_list
                        + trial.periphery_motion.to_list
                        + trial.seconds_on_periphery
                        + trial.seconds_on_center
                        + trial.periphery_entries
                        + trial.center_entries
                        for trial in self.trial_objects
                    ),
                    columns=(
                        *self._motion_2d_multi_indexer("Periphery"),
                        *self._motion_2d_multi_indexer("Center"),
                        *self._feature_2d_multi_indexer("Time_spent", periphery_center),
                        *self._feature_2d_multi_indexer("Entries", periphery_center),
                    ),
                    index=self._frame_index,
                ),
            ),
            axis=1,
        )


class SquareEnclosedTrial(BaseTrial):
    def __init__(
        self,
        center_metric_length: Union[float, int],
        **base_trial_kwargs,
    ):
        """
        Parameters
        ----------
        metric_resolution: float

        center_metric_length: float
            Length of the square box signifying periphery and inner area of the
            square box
        base_trial_kwargs
            Keyword arguments passed to BaseTrial
        """
        super().__init__(**base_trial_kwargs)

        self.center_metric_length = float(center_metric_length)
        assert self.metric_resolution > self.center_metric_length

    @cached_property
    def center_square_corners(self):
        def non_square_rectification(x_bias: float = 0.0, y_bias: float = 0.0):
            if (x_bias := float(x_bias)) and (y_bias := float(y_bias)):
                raise ValueError

            if x_bias:
                y_short = self.vertical_resolution * center_box_ratio
                y_long = self.vertical_resolution * one_minus_center_box_ratio

                x_short = y_short + x_bias
                x_long = y_long + x_bias

            else:
                x_short = self.horizontal_resolution * center_box_ratio
                x_long = self.horizontal_resolution * one_minus_center_box_ratio

                y_short = x_short + y_bias
                y_long = x_long + y_bias

            return np.array(
                (
                    (x_short, y_short),
                    (x_short, y_long),
                    (x_long, y_long),
                    (x_long, y_short),
                )
            )

        center_box_ratio = (
            (self.metric_resolution - self.center_metric_length) / 2.0
        ) / self.metric_resolution
        one_minus_center_box_ratio = 1.0 - center_box_ratio

        x = self.horizontal_resolution * center_box_ratio
        x_rest_half = (self.horizontal_resolution - x) / 2.0

        y = self.vertical_resolution * center_box_ratio
        y_rest_half = (self.vertical_resolution - y) / 2.0
        if self.horizontal_resolution == self.vertical_resolution:
            return np.array(
                (
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
            )

        elif self.horizontal_resolution < self.vertical_resolution:
            return non_square_rectification(
                y_bias=(self.vertical_resolution - self.horizontal_resolution) / 2.0
            )

        else:
            return non_square_rectification(
                x_bias=(self.horizontal_resolution - self.vertical_resolution) / 2.0
            )

    # Center Periphery

    @cached_property
    def center_boolean_index(self):
        return points_in_parallelogram(
            self.center_square_corners[0],
            self.center_square_corners[3],
            self.center_square_corners[1],
            self.coordinates_per_frame,
            inspect=self.inspection_figure_save,
            inspect_function_call_context=self.__class__.__name__,
        )

    @cached_property
    def periphery_boolean_index(self):
        return ~self.center_boolean_index

    @cached_property
    def location_sequence_center_periphery(self):
        # 1 is center, 2 is periphery, 0 is unknown
        location_sequence_center_periphery = np.zeros_like(
            self.center_boolean_index, dtype=np.uint8
        )
        location_sequence_center_periphery[self.center_boolean_index] = 1
        location_sequence_center_periphery[self.periphery_boolean_index] = 2
        return np.array(
            reduce_repeating_sequences(
                location_sequence_center_periphery,
                frame_tolerance=self._frame_tolerance,
            )
        )

    @cached_property
    def center_entries(self):
        return np.sum(self.location_sequence_center_periphery == 1)

    @cached_property
    def periphery_entries(self):
        return np.sum(self.location_sequence_center_periphery == 2)

    @cached_property
    def seconds_on_center(self):
        return np.sum(self.center_boolean_index) / self.fps

    @cached_property
    def seconds_on_periphery(self):
        return np.sum(self.periphery_boolean_index) / self.fps

    @cached_property
    def center_freezing_time(self):
        return (
            np.sum(self.frozen_boolean_index & self.center_boolean_index[1:]) / self.fps
        )

    @cached_property
    def periphery_freezing_time(self):
        return (
            np.sum(self.frozen_boolean_index & self.periphery_boolean_index[1:])
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
            self.motion.to_list
            + self.periphery_motion.to_list
            + self.center_motion.to_list
        )

    @property
    def feature_info(self):
        return [
            self.seconds_on_periphery,
            self.seconds_on_center,
            self.periphery_entries,
            self.center_entries,
        ]

    @property
    def info(self):
        return super().info + self.motion_info + self.feature_info
