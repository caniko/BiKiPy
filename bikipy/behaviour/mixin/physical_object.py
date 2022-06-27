from functools import cached_property, lru_cache
from typing import Any, ClassVar, Optional, TypeVar

import numpy as np
import pandas as pd
from pydantic import BaseModel, DirectoryPath, Field

from bikipy.behaviour.base import BaseTrial
from bikipy.behaviour.rectangle import RectangleEnclosedTrial
from bikipy.core.base_class import BaseBikipy
from bikipy.core.typing import NDArrayFp64
from bikipy.feature.physical_object import PhysicalObjectSet


class PhysicalObjectTrialMixin(BaseBikipy):
    gaze_start_point_label: Optional[str] = Field(description="Label of the eye center in the df")
    gaze_travel_direction_point_label: Optional[str] = Field(
        description="Label signifying the area where the gaze vector"
    )
    perimeter_border_normal_meters: Optional[float] = Field(
        description="The magnitude of the normal between the perimeter and the perimeter given in meters",
    )

    maximum_radians_inter_gaze_perimeter: float = 1.0 / 3.0 * np.pi
    minimum_seconds_attention: float = 1.0 / 3.0
    maximum_seconds_distraction: float = 2.0 / 3.0

    physical_object_inspect: bool = False

    physical_object_labels: ClassVar[list[str, ...]] = []
    all_perimeters_are_physical_objects: ClassVar[bool] = True

    @classmethod
    @property
    def number_of_physical_objects(cls) -> int:
        return len(cls.physical_object_labels)

    @classmethod
    @property
    def feature_headers(cls) -> list[tuple[str, ...]]:
        return list(pd.MultiIndex.from_product([["SecondsObserving"], ["All", *cls.physical_object_labels]]))

    @cached_property
    def perimeters(self):
        return self.all_physical_object_perimeters

    @property
    def feature_df_rows(self) -> list:
        return [
            self.physical_object_set.seconds_observing,
            *self.physical_object_set.object_specific_observation.values(),
        ]

    @cached_property
    def physical_object_keyword_arguments(self) -> dict[str, Any]:
        result = {
            "reader": self.reader,
            "gaze_travel_direction_point_label": self.gaze_travel_direction_point_label,
            "gaze_start_point_label": self.gaze_start_point_label,
            "maximum_radians_inter_gaze_perimeter": self.maximum_radians_inter_gaze_perimeter,
            "minimum_seconds_attention": self.minimum_seconds_attention,
            "maximum_seconds_distraction": self.maximum_seconds_distraction,
            "perimeter_border_normal_meters": self.perimeter_border_normal_meters,
        }

        if self.physical_object_inspect:
            if not self.inspect_directory:
                msg = "Physical object inspection is set to True, yet inspect_directory is undefined"
                raise AttributeError(msg)
            result["inspect_figure_file_path"] = (
                _physical_object_inspection_dir(self.inspect_directory) / f"{self.label}.png"
            )
            result["inspect_image"] = self.inspect_image

        return result

    @cached_property
    def physical_object_set(self) -> PhysicalObjectSet:
        return PhysicalObjectSet.from_perimeter(
            *self.all_physical_object_perimeters, **self.physical_object_keyword_arguments
        )

    @cached_property
    def perimeter_border_normal_pixels(self) -> float | NDArrayFp64:
        return self.perimeter_border_normal_meters * self.video.pixels_per_meter


PhysicalObjectTrial = TypeVar("PhysicalObjectTrial", bound=PhysicalObjectTrialMixin)


class PhysicalObjectHabituationTrialMixin(BaseModel):
    """The purpose of this stage is to generate reference data for proceeding experiments with objects."""

    trial_label: ClassVar[str] = "Habituation"


class RectangleEnclosedPhysicalObjectTrial(PhysicalObjectTrialMixin, RectangleEnclosedTrial):
    pass


@lru_cache(1)
def _physical_object_inspection_dir(global_inspection_dir: DirectoryPath) -> DirectoryPath:
    result = global_inspection_dir / "physical_object"
    result.mkdir(exist_ok=True)
    return result
