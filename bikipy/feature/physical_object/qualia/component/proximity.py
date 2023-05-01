from functools import cached_property
from typing import Optional, Hashable

from pydantic_numpy import NDArrayBool

from bikipy.feature.attention.proximity import proximity_filter
from bikipy.feature.physical_object.qualia.component.abc import AbstractQualiaComponent
from bikipy.utils.math.cached import meters2pixels


class Proximity(AbstractQualiaComponent):
    proximal_label: Hashable = ...
    outside_perimeter_border_label: Optional[Hashable]
    outside_label: Optional[Hashable]

    maximum_distance_meters: float = 0.05

    @cached_property
    def boolean_index(self) -> NDArrayBool:
        return proximity_filter(
            self.perimeter,
            meters2pixels(self.maximum_distance_meters, self.video.pixels_per_meter),
            self.reader[self.proximal_label],
            outside_perimeter_border=self.reader[self.outside_perimeter_border_label]
            if self.outside_perimeter_border_label
            else None,
            outside_perimeter=self.reader[self.outside_label] if self.outside_label else None,
            manual_ax=self.ax,
            **self._global_attention_kwargs,
        )
