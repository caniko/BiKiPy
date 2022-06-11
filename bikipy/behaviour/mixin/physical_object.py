from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional

import numpy as np
from pydantic import Field

from bikipy.core.base_class import BikipyBase
from bikipy.feature.physical_object.core import PhysicalObjectSet
from bikipy.perimeter.typing import AnyPerimeter


class PhysicalObjectBaseMixin(BikipyBase):
    gaze_start_point_label: str = Field(description="Label of the eye center in the df")
    gaze_travel_direction_point_label: str = Field(description="Label signifying the area where the gaze vector")
    perimeter_border_normal_metric_magnitude: Optional[float] = Field(
        None,
        description="The magnitude of the normal between the perimeter and the perimeter given in meters",
    )
    maximum_radians_inter_gaze_perimeter: float = 1 / 3 * np.pi
    minimum_seconds_attention: float = 0.5
    maximum_seconds_distraction: float = 0.5

    @cached_property
    def perimeter_border_normal_pixel_magnitude(self):
        return self.perimeter_border_normal_metric_magnitude / np.mean(self.meters_per_pixel)

    @cached_property
    def _physical_object_keyword_arguments(self):
        return {
            "fps": self.fps,
            "perimeter_border_normal_pixel_magnitude": self.perimeter_border_normal_pixel_magnitude,
            "gaze_travel_direction_point_label": self.gaze_travel_direction_point_label,
            "gaze_start_point_label": self.gaze_start_point_label,
            "maximum_radians_inter_gaze_perimeter": self.maximum_radians_inter_gaze_perimeter,
            "minimum_seconds_attention": self.minimum_seconds_attention,
            "maximum_seconds_distraction": self.maximum_seconds_distraction,
            "inspection_dir": self.inspection_dir,
        }


class PhysicalObjectExperimentMixin(PhysicalObjectBaseMixin, ABC):
    pass


class PhysicalObjectTrialMixin(PhysicalObjectBaseMixin, ABC):
    @cached_property
    @abstractmethod
    def all_physical_object_perimeters(self) -> tuple[AnyPerimeter, ...]:
        ...

    @cached_property
    def number_of_physical_objects(self) -> int:
        return len(self.all_physical_object_perimeters)

    @cached_property
    def physical_object_labels(self):
        return tuple(perimeter.label for perimeter in self.all_physical_object_perimeters)

    @cached_property
    def physical_object_set(self) -> PhysicalObjectSet:
        return PhysicalObjectSet.from_perimeter(
            *self.all_physical_object_perimeters, **self._physical_object_keyword_arguments
        )

    @cached_property
    def _physical_object_keyword_arguments(self):
        result = super()._physical_object_keyword_arguments
        result["reader"] = self.reader
        return result
