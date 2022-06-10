from functools import cached_property
from logging import getLogger
from typing import Any, Hashable, Optional

import numpy as np
from matplotlib import pyplot as plt
from pydantic import Field

from bikipy.behaviour.rectangle.rectangle import (
    RectangleEnclosedExperiment,
    RectangleEnclosedTrial,
)

logger = getLogger(__name__)


class SquareEnclosedExperiment(RectangleEnclosedExperiment):
    global_center_metric_length: Optional[float] = None

    def trial_keyword_arguments(self, trial_id: Hashable) -> dict:
        result = super().trial_keyword_arguments(trial_id)

        if self.global_center_metric_length:
            result["center_metric_length"] = self.global_center_metric_length

        return result


class SquareEnclosedTrial(RectangleEnclosedTrial):
    center_metric_length: Optional[float] = Field(
        description="Length of the square box signifying periphery and inner area " "of the square box"
    )

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

        center_box_ratio = ((self.metric_resolution - self.center_metric_length) / 2.0) / self.metric_resolution
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
            return non_square_rectification(y_bias=(self.vertical_resolution - self.horizontal_resolution) / 2.0)

        else:
            return non_square_rectification(x_bias=(self.horizontal_resolution - self.vertical_resolution) / 2.0)

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
    def motion_features(self):
        return super().motion_features + [
            *self.motion_center.values(),
            *self.motion_periphery.values(),
            self.seconds_on_periphery,
            self.seconds_on_center,
            self.periphery_entries,
            self.center_entries,
        ]
