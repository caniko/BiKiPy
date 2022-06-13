from abc import ABC, abstractmethod
from functools import cached_property
from typing import ClassVar, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, DirectoryPath, Field

from bikipy.behaviour.base import BaseExperiment, BaseTrial
from bikipy.behaviour.rectangle import (
    RectangleEnclosedExperiment,
    RectangleEnclosedTrial,
)
from bikipy.core.base_class import BikipyBase
from bikipy.feature.physical_object.core import PhysicalObjectSet
from bikipy.perimeter.typing import AnyPerimeter


class PhysicalObjectBaseMixin(BikipyBase, ABC):
    gaze_start_point_label: Optional[str] = Field(description="Label of the eye center in the df")
    gaze_travel_direction_point_label: Optional[str] = Field(
        description="Label signifying the area where the gaze vector"
    )
    perimeter_border_normal_metric_magnitude: Optional[float] = Field(
        description="The magnitude of the normal between the perimeter and the perimeter given in meters",
    )

    maximum_radians_inter_gaze_perimeter: float = 1 / 3 * np.pi
    minimum_seconds_attention: float = 0.5
    maximum_seconds_distraction: float = 0.5

    inspection_dir: Optional[DirectoryPath] = None

    @property
    @abstractmethod
    def video_metadata_can_be_defined(self) -> bool:
        ...

    @property
    @abstractmethod
    def meters_per_pixel(self):
        ...

    @property
    @abstractmethod
    def fps(self) -> float:
        ...

    @cached_property
    def perimeter_border_normal_pixel_magnitude(self):
        return self.perimeter_border_normal_metric_magnitude / self.meters_per_pixel

    @cached_property
    def _physical_object_keyword_arguments(self):
        result = {
            "gaze_travel_direction_point_label": self.gaze_travel_direction_point_label,
            "gaze_start_point_label": self.gaze_start_point_label,
            "maximum_radians_inter_gaze_perimeter": self.maximum_radians_inter_gaze_perimeter,
            "minimum_seconds_attention": self.minimum_seconds_attention,
            "maximum_seconds_distraction": self.maximum_seconds_distraction,
            "inspection_dir": self.inspection_dir,
        }

        if self.video_metadata_can_be_defined:
            result["perimeter_border_normal_pixel_magnitude"] = self.perimeter_border_normal_pixel_magnitude

        try:
            result["fps"] = self.fps
        except AttributeError:
            pass

        return result


class PhysicalObjectExperimentMixin(PhysicalObjectBaseMixin, ABC):
    pass


class PhysicalObjectTrialMixin(PhysicalObjectBaseMixin, ABC):
    gaze_start_point_label: str = Field(..., description="Label of the eye center in the df")
    gaze_travel_direction_point_label: str = Field(..., description="Label signifying the area where the gaze vector")
    perimeter_border_normal_metric_magnitude: float = Field(
        ...,
        description="The magnitude of the normal between the perimeter and the perimeter given in meters",
    )

    physical_object_labels: ClassVar[list[str, ...]] = []

    @cached_property
    @abstractmethod
    def all_physical_object_perimeters(self) -> tuple[AnyPerimeter, ...]:
        ...

    @classmethod
    @property
    def number_of_physical_objects(cls):
        return len(cls.physical_object_labels)

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


class RectanglePhysicalObjectExperiment(BaseExperiment, PhysicalObjectExperimentMixin):
    pass


class RectanglePhysicalObjectTrial(BaseTrial, PhysicalObjectTrialMixin, ABC):
    @classmethod
    @property
    def feature_headers(cls) -> list[str]:
        return list(pd.MultiIndex.from_product([["SecondsObserving"], ["All", *cls.physical_object_labels]]))

    @property
    def feature_summary_row(self):
        return [
            self.physical_object_set.seconds_observing,
            *self.physical_object_set.object_specific_observation.values(),
        ]


class PhysicalObjectHabituationTrialMixin(BaseModel):
    """The purpose of this stage is to generate reference data for proceeding experiments with objects."""

    experiment_sequence_index: ClassVar[Optional[int]] = 0
    trial_label: ClassVar[str] = "Habituation"
