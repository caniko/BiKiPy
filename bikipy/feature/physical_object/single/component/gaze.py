from functools import cached_property

import numpy as np
import pandas as pd
from pydantic_numpy import NDArrayBool

from bikipy.feature.attention.proximity import proximity_filter
from bikipy.feature.physical_object.single.component.abc import AbcObservationComponent
from bikipy.utils.math.cached import cached_deg2rad


class GazeComponent(AbcObservationComponent):
    center_eye_label: str = "center_eye"
    left_eye_label: str = "left_ear"
    right_eye_label: str = "right_ear"

    maximum_gaze_distance_meters: float = 0.05
    gaze_maximum_degrees: float = 45.0

    native_inspection_row_length = 4
    component_label = "Gaze"

    @cached_property
    def gaze_length_pixels(self) -> float:
        return self.maximum_gaze_distance_meters * self.video.pixels_per_meter

    @cached_property
    def gaze_maximum_radians(self) -> float:
        return cached_deg2rad(self.gaze_maximum_degrees)

    @cached_property
    def left_proximity(self) -> NDArrayBool:
        return proximity_filter(
            self.perimeter,
            self.reader[self.left_eye_label],
            self.gaze_length_pixels,
            self.reader[self.left_eye_label],
            manual_ax=self.axes_row[0],
            **self._global_attention_kwargs,
        )

    @cached_property
    def leftward_observation(self) -> NDArrayBool:
        result = np.zeros_like(self.left_proximity, dtype=bool)
        result[self.left_proximity] = self.perimeter.ray_direction_filter(
            self.reader[self.left_eye_label][self.left_proximity],
            self.reader[self.center_eye_label][self.left_proximity],
            self.gaze_maximum_radians,
            manual_ax=self.axes_row[1],
            **self._global_attention_kwargs,
        )
        return result

    @cached_property
    def right_proximity(self) -> NDArrayBool:
        return proximity_filter(
            self.perimeter,
            self.reader[self.right_eye_label],
            self.gaze_length_pixels,
            self.reader[self.right_eye_label],
            manual_ax=self.axes_row[2],
            **self._global_attention_kwargs,
        )

    @cached_property
    def rightward_observation(self) -> NDArrayBool:
        result = np.zeros_like(self.right_proximity, dtype=bool)
        result[self.right_proximity] = self.perimeter.ray_direction_filter(
            self.reader[self.right_eye_label][self.right_proximity],
            self.reader[self.center_eye_label][self.right_proximity],
            self.gaze_maximum_radians,
            manual_ax=self.axes_row[3],
            **self._global_attention_kwargs,
        )
        return result

    @cached_property
    def combined_sensation(self) -> NDArrayBool:
        return self.leftward_observation | self.rightward_observation

    @property
    def component_summary(self) -> pd.Series:
        return pd.concat(
            [
                super().component_summary,
                pd.Series(
                    [
                        self.boolean_array_to_seconds(self.left_proximity),
                        self.boolean_array_to_seconds(self.leftward_observation),
                        self.boolean_array_to_seconds(self.right_proximity),
                        self.boolean_array_to_seconds(self.rightward_observation),
                    ],
                    index=self._summary_indexer(
                        ["LeftProximity", "LeftwardObservation", "RightProximity", "RightwardObservation"]
                    ),
                ),
            ]
        )
