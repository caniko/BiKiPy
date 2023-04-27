from functools import cached_property

from pydantic_numpy import NDArrayBool

from bikipy.feature.attention.proximity import proximity_filter
from bikipy.feature.physical_object.single.component.abc import AbcObservationComponent


class OlfactionComponent(AbcObservationComponent):
    """Olfaction should only consist of a proximity element, and shouldn't need any ray casting"""

    nose_label: str = "nose"
    maximum_olfaction_distance_meters: float = 0.03

    native_inspection_row_length = 0
    component_label = "Olfaction"

    @property
    def maximum_olfaction_distance_pixels(self) -> float:
        return self.maximum_olfaction_distance_meters * self.video.pixels_per_meter

    @cached_property
    def nose_proximity(self):
        return proximity_filter(
            self.perimeter,
            self.reader[self.nose_label],
            self.maximum_olfaction_distance_pixels,
            manual_ax=self.axes_row[0],
            **self._global_attention_kwargs,
        )

    @property
    def combined_sensation(self) -> NDArrayBool:
        return self.nose_proximity

    @property
    def component_summary_dict(self) -> dict:
        pass
