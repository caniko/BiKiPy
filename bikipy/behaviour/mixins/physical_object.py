from abc import abstractproperty, ABC
from functools import cached_property

import numpy as np
from pydantic import BaseModel, Field

from bikipy.feature.physical_object import PhysicalObjectSet, PhysicalObject


class PhysicalObjectExperimentMixin(BaseModel, ABC):
    gaze_travel_direction_point_label: str = Field(
        description="Label signifying the area where the gaze vector"
    )
    gaze_start_point_label: str = Field(description="Label of the eye center in the df")
    perimeter_border_normal_metric_magnitude: float = Field(
        description="The magnitude of the normal between the perimeter and the border given in meters",
    )
    maximum_radians_inter_gaze_perimeter: float = 1 / 4 * np.pi
    minimum_seconds_attention: float = 0.5

    @abstractproperty
    def units_per_pixel(self):
        """Used to compute the metric distance from pixel values, and vice versa"""
        ...

    @abstractproperty
    def reader(self):
        ...

    @abstractproperty
    def fps(self):
        ...

    @abstractproperty
    def inspection_figure_save(self):
        ...

    @cached_property
    def perimeter_border_normal_pixel_magnitude(self):
        return self.perimeter_border_normal_metric_magnitude / np.mean(
            self.units_per_pixel
        )

    @cached_property
    def _physical_object_keyword_arguments(self):
        try:
            return {
                "reader": self.reader,
                "gaze_travel_direction_point_label": self.gaze_travel_direction_point_label,
                "gaze_start_point_label": self.gaze_start_point_label,
                "fps": self.fps,
                "perimeter_border_normal_pixel_magnitude": self.perimeter_border_normal_pixel_magnitude,
                "maximum_radians_inter_gaze_perimeter": self.maximum_radians_inter_gaze_perimeter,
                "minimum_seconds_attention": self.minimum_seconds_attention,
                "inspect": self.inspection_figure_save,
            }
        except AttributeError as e:
            msg = "The class does not support instancing PhysicalObject"
            raise NotImplementedError(msg) from e


class PhysicalObjectTrialMixin(BaseModel):
    physical_object_set: PhysicalObjectSet

    @classmethod
    def from_perimeter(cls, *perimeters, **kwargs):
        physical_objects = PhysicalObjectSet(tuple(
            PhysicalObject(perimeter, **kwargs) for perimeter in perimeters
        ))
        return cls(physical_object_set=physical_objects)
