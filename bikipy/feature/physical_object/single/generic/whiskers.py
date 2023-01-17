from functools import cached_property
from typing import ClassVar

import numpy as np
from pydantic_numpy import NDArrayBool

from bikipy.feature.attention.proximity import proximity_filter
from bikipy.feature.physical_object.single.component import AbcObservationComponent


class WhiskerInteraction(AbcObservationComponent):
    nose_label: str
    left_eye_label: str
    right_eye_label: str

    whisker_length_meters: float = 0.4

    native_inspection_row_length = 4

    whisker_midpoint_distance_from_nose_to_eye: ClassVar[float] = 0.2
    left_whisker_label: ClassVar[str] = "left_whisker"
    right_whisker_label: ClassVar[str] = "right_whisker"

    @cached_property
    def whisker_length_pixels(self) -> float:
        return self.whisker_length_meters * self.video.pixels_per_meter

    @cached_property
    def left_proximity(self) -> NDArrayBool:
        return proximity_filter(
            self.perimeter,
            self.reader[self.left_label].values,
            self.whisker_length_pixels,
            self.reader[self.origin_point_label].values,
            manual_ax=self.axes_row[0],
            **self._global_attention_kwargs,
        )

    @cached_property
    def left_whisker_ray(self) -> NDArrayBool:
        self.reader.add_midpoint(
            self.left_eye_label,
            self.nose_label,
            self.left_whisker_label,
            midpoint_multiplier=self.whisker_midpoint_distance_from_nose_to_eye,
        )
        result = np.zeros_like(self.left_proximity, dtype=bool)
        result[self.left_proximity] = self.perimeter.ray_direction_filter(
            self._gaze_travel_direction_point,
            self.reader[self.left_whisker_label].values,
            self.maximum_radians_inter_gaze_perimeter,
            manual_ax=self.axes_row[1],
            **self._global_attention_kwargs,
        )
        return result

    @cached_property
    def right_proximity(self) -> NDArrayBool:
        return proximity_filter(
            self.perimeter,
            self.reader[self.right_label].values,
            self.whisker_length_pixels,
            self.reader[self.origin_point_label].values,
            manual_ax=self.axes_row[2],
            **self._global_attention_kwargs,
        )

    @property
    def combined_sensation(self) -> NDArrayBool:
        pass
