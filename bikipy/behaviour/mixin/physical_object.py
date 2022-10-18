from functools import cached_property, lru_cache
from typing import Any, ClassVar, Optional, TypeVar

import numpy as np
import pandas as pd
from pydantic import DirectoryPath, Field
from pydantic_numpy.dtype import NDArrayFp64

from bikipy.behaviour.core.enclosure.rectangle import RectangleEnclosedTrial
from bikipy.core.base_class import BaseBikipy
from bikipy.feature.physical_object import PhysicalObjectSet
from bikipy.feature.tolerance import (
    GENERIC_MAXIMUM_SECONDS_DISTRACTION,
    GENERIC_MINIMUM_SECONDS_ATTENTION,
)


class PhysicalObjectTrialMixin(BaseBikipy):
    perimeter_border_normal_meters: float = Field(
        ...,
        description="The magnitude of the normal between the perimeter and the perimeter given in meters",
    )
    gaze_start_point_label: str = Field(
        ..., description="Label of the eye center (or the origin of the gaze) in the df"
    )
    gaze_travel_direction_point_label: str = Field(
        ..., description="Label signifying the area where the gaze vector will be cast"
    )

    outside_perimeter_point_label: Optional[str] = Field(description="Label signifying the area where the gaze vector")

    maximum_radians_inter_gaze_perimeter: float = np.pi / 4.0
    minimum_seconds_attention: float = GENERIC_MINIMUM_SECONDS_ATTENTION
    maximum_seconds_distraction: float = GENERIC_MAXIMUM_SECONDS_DISTRACTION

    physical_object_inspect: bool = False

    physical_object_labels: ClassVar[tuple[str, ...]] = ...
    all_perimeters_are_physical_objects: ClassVar[bool] = True

    @classmethod
    @property
    def number_of_physical_objects(cls) -> int:
        return len(cls.physical_object_labels)

    @cached_property
    def perimeters(self):
        return self.all_physical_object_perimeters

    @cached_property
    def trial_feature_series(self):
        return pd.concat(
            (*self._trial_physical_object_feature_series_list[::-1], *self._trial_feature_series_list[::-1]), axis=0
        )

    @cached_property
    def _trial_physical_object_feature_series_list(self) -> list[pd.Series, ...]:
        return [self.physical_object_set.feature_summary]

    @cached_property
    def physical_object_keyword_arguments(self) -> dict[str, Any]:
        return {
            "reader": self.reader,
            "trial_obj_label": self.label,
            "gaze_travel_direction_point_label": self.gaze_travel_direction_point_label,
            "gaze_start_point_label": self.gaze_start_point_label,
            "outside_perimeter_point_label": self.outside_perimeter_point_label,
            "maximum_radians_inter_gaze_perimeter": self.maximum_radians_inter_gaze_perimeter,
            "minimum_seconds_attention": self.minimum_seconds_attention,
            "maximum_seconds_distraction": self.maximum_seconds_distraction,
            "perimeter_border_normal_pixels": self.perimeter_border_normal_pixels,
            "inspect_arg": self.inspect_arg,
        }

    @cached_property
    def physical_object_set(self) -> PhysicalObjectSet:
        return PhysicalObjectSet.from_perimeter(
            *self.all_physical_object_perimeters, **self.physical_object_keyword_arguments
        )

    @cached_property
    def perimeter_border_normal_pixels(self) -> float | NDArrayFp64:
        return self.perimeter_border_normal_meters * self.video.pixels_per_meter


PhysicalObjectTrial = TypeVar("PhysicalObjectTrial", bound=PhysicalObjectTrialMixin)


class RectangleEnclosedPhysicalObjectTrial(PhysicalObjectTrialMixin, RectangleEnclosedTrial):
    pass


@lru_cache(1)
def _physical_object_inspection_dir(global_inspection_dir: DirectoryPath) -> DirectoryPath:
    result = global_inspection_dir / "physical_object"
    result.mkdir(exist_ok=True)
    return result
