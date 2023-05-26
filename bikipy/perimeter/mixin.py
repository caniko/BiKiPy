from abc import ABC, abstractmethod
from functools import cached_property
from logging import getLogger
from typing import Generic, Literal, Optional, TypeVarTuple

from pydantic import Field, PositiveInt
from pydantic.generics import GenericModel

from bikipy.core.base import BikipyModel
from bikipy.core.typing import Label
from bikipy.core.video import VideoMetadata
from bikipy.perimeter.base import Perimeter

logger = getLogger(__name__)


PerimeterInstances = TypeVarTuple("PerimeterInstances")


class TrialWithPerimeterMixin(GenericModel, Generic[Perimeter, *PerimeterInstances], BikipyModel, ABC):
    # Derive meters per pixel from perimeter
    # TODO: Put this logic in the backend by prioritizing preferred sources
    meters_per_pixel_from_perimeter_source: Literal["side", "diagonal", "diameter", "radius", None] = Field(
        None,
        description="The perimeter attribute that will be used to derive meters_per_pixel. Supported sources with "
        "respect to SinglePerimeter type:\n"
        "Polygon: To be decided\n"
        "Regular polygon (every side has equal length): side\n"
        "Rectangle: diagonal\n"
        "Circle: diameter, radius\n",
    )
    length_meters_of_meters_per_pixel_source: Optional[float]
    manual_perimeter_to_derive_meters_per_pixel: Optional[str]

    @property
    @abstractmethod
    def perimeters(self) -> tuple[Perimeter, *PerimeterInstances]:
        ...

    def __getitem__(self, item) -> Perimeter:
        return self._label_to_perimeter[item]

    def _validate_perimeters_object(self) -> None:
        if not self.perimeters:
            msg = "perimeters is not defined as an object variable, " "which is required for _int_id_to_perimeter"
            raise AttributeError(msg)

    @cached_property
    def _int_id_to_perimeter(self) -> dict[PositiveInt, Perimeter]:
        self._validate_perimeters_object()
        return {perimeter.int_id: perimeter for perimeter in self.perimeters}

    @cached_property
    def _label_to_perimeter(self) -> dict[Label, Perimeter]:
        self._validate_perimeters_object()
        return {perimeter.label: perimeter for perimeter in self.perimeters}

    @property
    def _video(self) -> VideoMetadata:
        video = super()._video

        if self.perimeters:
            # Considered making this logic optional; going with user-side discretion instead
            for perimeter in self.perimeters:
                video = VideoMetadata.join(perimeter.video, video, ignore_incongruity=True)

            for perimeter in self.perimeters:
                perimeter.manual_video = video

        return video
