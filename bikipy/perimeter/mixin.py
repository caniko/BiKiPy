from abc import ABC, abstractmethod
from functools import cached_property, reduce
from logging import getLogger
from typing import Generic, TypeVarTuple

from pydantic import PositiveInt
from pydantic.generics import GenericModel

from bikipy.core.base import BikipyModel
from bikipy.core.typing import Label
from bikipy.core.video import VideoMetadata, incongruity_permissive_video_join
from bikipy.perimeter.base import Perimeter

logger = getLogger(__name__)


PerimeterInstances = TypeVarTuple("PerimeterInstances")


class TrialWithPerimeterMixin(GenericModel, Generic[Perimeter, *PerimeterInstances], BikipyModel, ABC):
    @property
    @abstractmethod
    def perimeters(self) -> list[Perimeter, *PerimeterInstances]:
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
            perimeter_video = reduce(
                incongruity_permissive_video_join, (perimeter.video for perimeter in self.perimeters)
            )
            # Manually passed video parameters should override any
            new_video = VideoMetadata.join(video, perimeter_video, ignore_incongruity=True)

            # The resolution on perimeters should be more correct than whatever
            # provided by the user, hence it being master
            final_video = VideoMetadata.join(new_video, video, ignore_incongruity=True)

            if self.meters_per_pixel_from_perimeter:
                logger.debug("meters_per_pixel_from_perimeter -> True: Deriving meters_per_pixel from perimeter")
                if not self.perimeter_to_derive_meters_per_pixel:
                    msg = f"perimeter_to_derive_meters_per_pixel is not defined for class, {self.__class__.__name__}"
                    raise AttributeError(msg)

                final_video.meters_per_pixel = (
                    self.perimeter_to_derive_meters_per_pixel.derived_meters_per_pixel.derived_meters_per_pixel
                )

            for perimeter in self.perimeters:
                perimeter.manual_video = final_video
            return final_video

        return video
