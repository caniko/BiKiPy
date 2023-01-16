from functools import cached_property

from pydantic_numpy import NDArrayFp64, NDArrayBool

from bikipy.feature.attention.proximity import proximity_filter
from bikipy.feature.physical_object.single.component import AbcObservationComponent


class WhiskerInteraction(AbcObservationComponent):
    origin_point_label: str
    left_label: str
    right_label: str

    whisker_length_meters: float = 0.4

    native_inspection_row_length = 4

    @cached_property
    def left_proximity(self) -> NDArrayBool:
        return proximity_filter(
            self.perimeter,
            self.reader[self.left_label].values,
            self._outside_perimeter_point if self.outside_perimeter_point_label else self._gaze_start_point,
            self.perimeter_border_normal_pixels,
            manual_ax=self.proximity_ax,
            **self.physical_object._global_attention_kwargs,
        )

    @property
    def combined_sensation(self) -> NDArrayBool:
        pass
