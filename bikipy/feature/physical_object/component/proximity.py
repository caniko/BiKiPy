from functools import cached_property

from pydantic_numpy import NDArrayBool

from bikipy.feature.attention.proximity import proximity_filter
from bikipy.feature.physical_object.component.abc import AbcQualiaComponent


class Proximity(AbcQualiaComponent):
    proximal_label: str = ...

    maximum_distance_meters: float = 0.05

    native_inspection_row_length = 2
    component_label = "NoseTailProximity"

    @property
    def maximum_distance_pixels(self) -> float:
        return self.maximum_distance_meters * self.video.pixels_per_meter

    @cached_property
    def boo(self):
        return proximity_filter(
            self.perimeter,
            self.reader[self.proximal_label],
            self.maximum_distance_pixels,
            manual_ax=self.axes_row[0],
            **self._global_attention_kwargs,
        )

    @cached_property
    def tail_proximity(self) -> NDArrayBool:
        return proximity_filter(
            self.perimeter,
            self.reader[self.tail_label],
            self.maximum_distance_meters,
            manual_ax=self.axes_row[1],
            **self._global_attention_kwargs,
        )

    @cached_property
    def boolean_index(self) -> NDArrayBool:
        result = self.nose_proximity | self.tail_proximity
        self._combined_sensation_plot(result)
        return result

    @property
    def summary_series(self) -> dict:
        return {
            "NoseProximity": self.boolean_array_to_seconds(self.nose_proximity),
            "TailProximity": self.boolean_array_to_seconds(self.tail_proximity),
        }
