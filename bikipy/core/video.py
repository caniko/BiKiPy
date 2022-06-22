"""
Weird dataclass hierarchy for videos, requirement by design.

Purpose of VideoMetadataMixin
Classes that define videos should have this mixin: VideoMetadata, BaseExperiment, and BaseTrial. This class is
bare metadata, and its purpose is to either initialize or relay an existing VideoMetadata object
"""
from functools import cached_property
from logging import getLogger
from typing import Optional, Any, ClassVar

import numpy as np
from numpy import ndarray
from pydantic import FilePath

from bikipy.core.base_class import BikipyBase
from bikipy.core.typing import NDArrayFp64, NDArrayUint8, NDArrayInt16
from bikipy.utils.video import get_video_data

logger = getLogger(__name__)


class _VideoMetadataBase(BikipyBase):
    meters_per_pixel: Optional[NDArrayFp64]

    video_path: Optional[FilePath]
    manual_recording_resolution: Optional[NDArrayInt16]
    manual_fps: Optional[float]
    manual_frame: Optional[NDArrayUint8 | FilePath]


class VideoMetadata(_VideoMetadataBase):
    def __and__(self, other: "VideoMetadata") -> bool:
        for key in set(self.manual_video_metadata).intersection(other.manual_video_metadata):
            if np.any(self.manual_video_metadata[key] != other.manual_video_metadata[key]):
                logger.debug(
                    f"self and other are incongruent on {key}: "
                    f"{self.manual_video_metadata[key]} != {other.manual_video_metadata[key]}"
                )
                return False
        return True

    def __eq__(self, other: "VideoMetadata") -> bool:
        return set(self.manual_video_metadata) == set(other.manual_video_metadata)

    def __add__(self, other: "VideoMetadata") -> "VideoMetadata":
        return self.join(self, other)

    @classmethod
    def join(
        cls, master: "VideoMetadata", slave: "VideoMetadata", ignore_incongruency: bool = False
    ) -> "VideoMetadata":
        if not (master & slave) and not ignore_incongruency:
            msg = "VideoMetadata are incongruent"
            raise AttributeError(msg)
        new_metadata = slave.manual_video_metadata
        new_metadata.update(master.manual_video_metadata)
        return cls(**new_metadata)

    @cached_property
    def pixels_per_meter(self) -> NDArrayFp64:
        return 1.0 / self.meters_per_pixel

    @cached_property
    def recording_resolution(self) -> NDArrayInt16:
        return (
            self.manual_recording_resolution
            if self.manual_recording_resolution is not None
            else self._video_metadata_from_file[0]
        )

    @cached_property
    def center_pixel(self) -> NDArrayInt16:
        return np.round(self.recording_resolution / 2.0)

    @property
    def horizontal_resolution(self) -> int:
        return self.recording_resolution[0]

    @property
    def vertical_resolution(self) -> int:
        return self.recording_resolution[1]

    @cached_property
    def metric_resolution(self) -> NDArrayFp64:
        return self.recording_resolution * self.meters_per_pixel

    @cached_property
    def center_meters(self) -> NDArrayFp64:
        return self.metric_resolution / 2.0

    @property
    def metric_horizontal_resolution(self) -> int:
        return self.metric_resolution[0]

    @property
    def metric_vertical_resolution(self) -> int:
        return self.metric_resolution[1]

    @property
    def fps(self) -> float:
        return self.manual_fps or self._video_metadata_from_file[1]

    @property
    def frame(self) -> NDArrayUint8 | None:
        if self.manual_frame is not None:
            return self.manual_frame
        if self.video_path:
            return self._video_metadata_from_file[2]

    @cached_property
    def video_metadata(self):
        result = {}
        if self.meters_per_pixel is not None:
            result["meters_per_pixel"] = self.meters_per_pixel
        if self.recording_resolution is not None:
            result["recording_resolution"] = self.recording_resolution
        if self.fps:
            result["fps"] = self.fps
        if self.frame is not None:
            result["frame"] = self.frame
        return result

    @cached_property
    def manual_video_metadata(self):
        return {
            f"manual_{key}" if key != "meters_per_pixel" else key: value for key, value in self.video_metadata.items()
        }

    @cached_property
    def _video_metadata_from_file(self) -> tuple[None, None, None] | tuple[ndarray, Any, Any]:
        if not self.video_path:
            return None, None, None

        frame, horizontal_resolution, vertical_resolution, fps = get_video_data(self.video_path)

        return np.array((horizontal_resolution, vertical_resolution), dtype=np.int16), fps, frame


class VideoMetadataMixin(_VideoMetadataBase):
    manual_video: Optional[VideoMetadata]

    required_video_metadata_fields: ClassVar[set[str]] = set()

    @property
    def video(self) -> VideoMetadata:
        # TODO: computed_field validation
        if self.required_video_metadata_fields and (
            missing_fields := self.required_video_metadata_fields.difference(self._video.video_metadata)
        ):
            msg = (
                f"{self.__class__.__name__} requires {self.required_video_metadata_fields}, "
                f"but is missing {missing_fields}"
            )
            raise AttributeError(msg)
        return self._video

    @property
    def video_metadata(self):
        return self.video.video_metadata

    @cached_property
    def _video(self):
        if self.manual_video:
            return self.manual_video
        return VideoMetadata(
            video_path=self.video_path,
            meters_per_pixel=self.meters_per_pixel,
            manual_fps=self.manual_fps,
            manual_recording_resolution=self.manual_recording_resolution,
        )


def convert_meters_to_pixels(data: NDArrayFp64, video: VideoMetadata) -> NDArrayFp64:
    return data * video.pixels_per_meter


def inspect_video_is_none_during_inspection(inspect_video: VideoMetadata | None):
    if inspect_video is None:
        msg = (
            "inspect_video is required to map the result from pixels to meters; "
            "required for generating inspection figure"
        )
        raise ValueError(msg)
