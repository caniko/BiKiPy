from abc import ABC, abstractmethod
from functools import cached_property
from logging import getLogger
from typing import Literal, Optional

from pydantic import Field, PositiveInt, computed_field

from bikipy.core.base import BikipyModel
from bikipy.core.typing import Label
from bikipy.core.video import VideoMetadata
from bikipy.perimeter.base import BasePerimeter

logger = getLogger(__name__)


class TrialWithPerimeterMixin[Perimeter: BasePerimeter, *PerimeterInstances](BikipyModel, ABC):
    # Derive meters per pixel from perimeter
    # TODO: Put this logic in the backend by prioritizing preferred sources
    meters_per_pixel_from_perimeter_source: Literal["side", "diagonal", "diameter", "radius", None] = Field(
        None,
        description="The perimeter attribute that will be used to derive meters_per_pixel. Supported sources with "
        "respect to BaseSinglePerimeter type:\n"
        "Polygon: To be decided\n"
        "Regular polygon (every side has equal length): side\n"
        "Rectangle: diagonal\n"
        "Circle: diameter, radius\n",
    )
    length_meters_of_meters_per_pixel_source: Optional[float] = None
    manual_perimeter_to_derive_meters_per_pixel: Optional[str] = None

    @property
    @abstractmethod
    def perimeters(self) -> tuple[Perimeter, *PerimeterInstances]: ...

    @computed_field  # type: ignore[misc]
    @property
    def perimeter_to_derive_meters_per_pixel(self) -> Perimeter:
        if self.manual_perimeter_to_derive_meters_per_pixel:
            try:
                return self.label_to_perimeter[self.manual_perimeter_to_derive_meters_per_pixel]
            except TypeError:
                # self.label_to_perimeter is None -> TypeError
                msg = (
                    f"The class, {self.__class__.__name__}, does not define label_to_perimeter, "
                    f"which makes the mapping of manual_perimeter_to_derive_meters_per_pixel "
                    f"to a Perimeter object impossible"
                )
                raise AttributeError(msg)

    def __getitem__(self, item) -> Perimeter:
        return self.label_to_perimeter[item]

    def _validate_perimeters_object(self) -> None:
        if not self.perimeters:
            msg = "perimeters is not defined as an object variable, " "which is required for _int_id_to_perimeter"
            raise AttributeError(msg)

    @computed_field  # type: ignore[misc]
    @cached_property
    def _int_id_to_perimeter(self) -> dict[PositiveInt, Perimeter]:
        self._validate_perimeters_object()
        return {perimeter.int_id: perimeter for perimeter in self.perimeters}

    @computed_field  # type: ignore[misc]
    @cached_property
    def label_to_perimeter(self) -> dict[Label, Perimeter]:
        self._validate_perimeters_object()
        return {perimeter.label: perimeter for perimeter in self.perimeters}

    @computed_field  # type: ignore[misc]
    @property
    def video(self) -> VideoMetadata:
        video = super().video

        if self.perimeters:
            # Considered making this logic optional; going with user-side discretion instead
            for perimeter in self.perimeters:
                video = VideoMetadata.join(perimeter.video, video, ignore_incongruity=True)

            for perimeter in self.perimeters:
                perimeter.manual_video = video

        return video
