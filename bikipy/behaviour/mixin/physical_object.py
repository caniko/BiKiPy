from abc import ABC, abstractmethod
from functools import cached_property
from typing import Optional

import numpy as np
from pydantic import Field

from bikipy.core.base_class import BikipyBase
from bikipy.feature.physical_object import PhysicalObjectSet


class PhysicalObjectBaseMixin(BikipyBase, ABC):
    gaze_start_point_label: str = Field(description="Label of the eye center in the df")
    gaze_travel_direction_point_label: str = Field(
        description="Label signifying the area where the gaze vector"
    )
    perimeter_border_normal_metric_magnitude: Optional[float] = Field(
        None,
        description="The magnitude of the normal between the perimeter and the perimeter given in meters",
    )
    maximum_radians_inter_gaze_perimeter: float = 1 / 4 * np.pi
    minimum_seconds_attention: float = 0.5

    @property
    @abstractmethod
    def meters_per_pixel(self):
        """Used to compute the metric distance from pixel values, and vice versa"""
        ...

    @property
    @abstractmethod
    def video_metadata_can_be_defined(self):
        ...

    @cached_property
    def perimeter_border_normal_pixel_magnitude(self):
        return self.perimeter_border_normal_metric_magnitude / np.mean(
            self.meters_per_pixel
        )

    @cached_property
    def _physical_object_keyword_arguments(self):
        try:
            result = {
                "gaze_travel_direction_point_label": self.gaze_travel_direction_point_label,
                "gaze_start_point_label": self.gaze_start_point_label,
                "maximum_radians_inter_gaze_perimeter": self.maximum_radians_inter_gaze_perimeter,
                "minimum_seconds_attention": self.minimum_seconds_attention,
                "inspect": self.inspection_figure_save,
            }
        except AttributeError as e:
            msg = "The class does not support instancing PhysicalObject"
            raise NotImplementedError(msg) from e

        if self.video_metadata_can_be_defined:
            result["fps"] = self.fps
            result[
                "perimeter_border_normal_pixel_magnitude"
            ] = self.perimeter_border_normal_pixel_magnitude

        return result


class PhysicalObjectExperimentMixin(PhysicalObjectBaseMixin, ABC):
    pass


class PhysicalObjectTrialMixin(PhysicalObjectBaseMixin, ABC):
    @property
    @abstractmethod
    def physical_object_set(self) -> PhysicalObjectSet:
        ...

    @cached_property
    def _physical_object_keyword_arguments(self):
        result = super()._physical_object_keyword_arguments
        result["reader"] = self.reader
        return result
