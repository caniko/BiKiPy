from functools import cached_property
from typing import Any, Optional

from pydantic import root_validator
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64

from bikipy.feature.attention.proximity import proximity_filter
from bikipy.feature.physical_object.single.main import PhysicalObject

PROXIMITY_FIELDS = ("perimeter_border_normal_pixels", "proximity_ax")  # impenatrable: outside_perimeter_point_label
RAY_CAST_FIELDS = ("ray_start_point_label", "ray_travel_direction_point_label", "ray_maximum_radians", "ray_ax")


class ProximityRayCast(AbcAnimalProfile):
    physical_object: PhysicalObject

    # Proximity fields
    perimeter_border_normal_pixels: Optional[float | NDArrayFp64]
    outside_perimeter_point_label: str | None
    proximity_ax: Any

    # Ray cast fields
    ray_start_point_label: Optional[str]
    ray_travel_direction_point_label: Optional[str]
    ray_maximum_radians: Optional[float]
    ray_ax: Any

    @root_validator(pre=True)
    def proximity_or_ray(cls, values):
        if all(field in values for field in PROXIMITY_FIELDS) or all(field in values for field in RAY_CAST_FIELDS):
            msg = (
                f"Either proximity or ray cast fields have to be fined:\n"
                f"Proximity: {PROXIMITY_FIELDS}\nRay: {RAY_CAST_FIELDS}"
            )
            raise AttributeError(msg)
        return values

    @cached_property
    def proximity_boolean_index(self) -> NDArrayBool:
        return proximity_filter(
            self.physical_object.perimeter,
            self._ray_travel_direction_point,
            self.perimeter_border_normal_pixels,
            self._outside_perimeter_point if self.outside_perimeter_point_label else self._ray_start_point,
            manual_ax=self.proximity_ax,
            **self.physical_object._global_attention_kwargs,
        )

    @cached_property
    def ray_boolean_index(self) -> NDArrayBool:
        return self.physical_object.perimeter.ray_direction_filter(
            self._ray_travel_direction_point,
            self._ray_start_point,
            self.ray_maximum_radians,
            manual_ax=self.ray_ax,
            **self.physical_object._global_attention_kwargs,
        )

    @cached_property
    def raw_combined(self) -> NDArrayFp64:
        return self.proximity_boolean_index & self.ray_boolean_index

    @property
    def _outside_perimeter_point(self) -> NDArrayFp64:
        return self.physical_object.reader[self.outside_perimeter_point_label].values

    @property
    def _ray_start_point(self) -> NDArrayFp64:
        return self.physical_object.reader[self.ray_start_point_label].values

    @property
    def _ray_travel_direction_point(self) -> NDArrayFp64:
        return self.physical_object.reader[self.ray_travel_direction_point_label].values
