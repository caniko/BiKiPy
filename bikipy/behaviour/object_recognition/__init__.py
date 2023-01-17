from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from typing import Any, ClassVar, Optional, TypeVar

import numpy as np
import pandas as pd
from pydantic import DirectoryPath, Field
from pydantic_numpy.dtype import NDArrayFp64

from bikipy.behaviour.core.abstract import AbstractTrial
from bikipy.feature.physical_object.set import PhysicalObjectSet
from bikipy.perimeter.base import SinglePerimeter


class ObjectRecognitionTrialMixin(AbstractTrial, ABC):
    perimeter_border_normal_meters: float = Field(
        ...,
        description="The magnitude of the normal between the perimeter and the perimeter given in meters",
    )
    ray_start_point_label: str = Field(..., description="Label of the eye center (or the origin of the ray) in the df")
    ray_travel_direction_point_label: str = Field(
        ..., description="Label signifying the area where the ray vector will be cast"
    )

    outside_perimeter_point_label: Optional[str] = Field(description="Label signifying the area where the ray vector")

    maximum_degrees_inter_ray_perimeter: float = 45

    physical_object_inspect: bool = False

    physical_object_labels: ClassVar[set[str]] = set()
    all_perimeters_are_physical_objects: ClassVar[bool] = True

    @property
    @abstractmethod
    def all_physical_object_perimeters(self) -> tuple[SinglePerimeter, ...]:
        ...

    @classmethod
    @property
    def number_of_physical_objects(cls) -> int:
        return len(cls.physical_object_labels)

    @cached_property
    def perimeters(self):
        return self.all_physical_object_perimeters

    @property
    def _trial_feature_series_list(self) -> list[pd.Series]:
        upstream_list = super()._trial_feature_series_list
        upstream_list.append(self.physical_object_set.feature_summary)
        return upstream_list

    @cached_property
    def maximum_radians_inter_ray_perimeter(self) -> float:
        return np.deg2rad(self.maximum_degrees_inter_ray_perimeter)

    @cached_property
    def physical_object_keyword_arguments(self) -> dict[str, Any]:
        return {
            "reader": self.reader,
            "trial_obj_label": self.label,
            "ray_travel_direction_point_label": self.ray_travel_direction_point_label,
            "ray_start_point_label": self.ray_start_point_label,
            "outside_perimeter_point_label": self.outside_perimeter_point_label,
            "maximum_radians_inter_ray_perimeter": self.maximum_radians_inter_ray_perimeter,
            "perimeter_border_normal_pixels": self.perimeter_border_normal_pixels,
            "inspect_arg": self.inspect_arg,
            "manual_video": self.video,
        }

    @cached_property
    def physical_object_set(self) -> PhysicalObjectSet:
        return PhysicalObjectSet.from_perimeter(
            *self.all_physical_object_perimeters, **self.physical_object_keyword_arguments
        )

    @cached_property
    def perimeter_border_normal_pixels(self) -> float | NDArrayFp64:
        return self.perimeter_border_normal_meters * self.video.pixels_per_meter


PhysicalObjectTrial = TypeVar("PhysicalObjectTrial", bound=ObjectRecognitionTrialMixin)


@lru_cache(1)
def _physical_object_inspection_dir(global_inspection_dir: DirectoryPath) -> DirectoryPath:
    result = global_inspection_dir / "physical_object"
    result.mkdir(exist_ok=True)
    return result
