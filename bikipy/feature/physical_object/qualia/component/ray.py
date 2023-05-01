from functools import cached_property

import numpy as np
from pydantic_numpy import NDArrayBool

from bikipy.feature.physical_object.qualia.component.abc import AbstractQualiaComponent
from bikipy.utils.math.cached import cached_deg2rad


class RayCasting(AbstractQualiaComponent):
    ray_origin_label: str = ...
    ray_direction_label: str = ...

    gaze_maximum_degrees: float = 45.0

    @cached_property
    def boolean_index(self) -> NDArrayBool:
        return self.perimeter.ray_direction_filter(
            self.reader[self.ray_direction_label],
            self.reader[self.ray_origin_label],
            cached_deg2rad(self.gaze_maximum_degrees),
            manual_ax=self.ax,
            **self._global_attention_kwargs,
        )
