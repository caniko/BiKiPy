"""
Weird dataclass hierarchy for videos, requirement by design.

Purpose of VideoMetadataMixin
Classes that define videos should have this mixin: VideoMetadata, BaseExperiment, and BaseTrial. This class is
barebones metadata, and its purpose is to either initialize or relay an existing VideoMetadata object
"""
from functools import cached_property
from typing import Optional

import numpy as np
from pydantic import FilePath

from bikipy.core.base_class import BikipyBase
from bikipy.core.typing import NDArrayFp64, NDArrayUint8, NDArrayInt16
from bikipy.utils.video import get_video_data


class _VideoMetadataBase(BikipyBase):
    video_path: Optional[FilePath]

    manual_recording_resolution: Optional[NDArrayInt16]
    manual_fps: Optional[float]

    manual_meters_per_pixel: Optional[NDArrayFp64 | float]
    metric_resolution: Optional[NDArrayFp64]


class VideoMetadata(_VideoMetadataBase):
    @cached_property
    def video_metadata(self):
        return {
            "manual_meter_per_pixel": self.meters_per_pixel if self.video_metadata_can_be_defined else None,
            "manual_fps": self.fps,
            "manual_recording_resolution": self.recording_resolution,
            "metric_resolution": self.metric_resolution,
        }

    @cached_property
    def video_metadata_can_be_defined(self) -> bool:
        return bool(
            self.manual_fps is not None
            and self.manual_recording_resolution is not None
            and (self.manual_meters_per_pixel is not None or self.metric_resolution is not None)
            or (self.video_path and (self.manual_meters_per_pixel is not None or self.metric_resolution is not None))
        )

    @cached_property
    def meters_per_pixel(self):
        if self.manual_meters_per_pixel is not None:
            return self.manual_meters_per_pixel

        if self.metric_resolution is None:
            msg = (
                "metric_resolution attribute needs to be defined to compute meters_per_pixel. "
                "Alternatively, you may define manual_meters_per_pixel; manual_fps, manual_recording_resolution "
                "still must be defined"
            )
            raise AttributeError(msg)

        return self.metric_resolution / self.recording_resolution

    @cached_property
    def pixels_per_meter(self):
        return 1.0 / self.meters_per_pixel

    @cached_property
    def recording_resolution(self) -> NDArrayInt16:
        return (
            self.manual_recording_resolution
            if self.manual_recording_resolution is not None
            else self._video_metadata_from_file[0]
        )

    @cached_property
    def tuple_recording_resolution(self) -> tuple:
        return tuple(self.recording_resolution)

    @property
    def horizontal_resolution(self) -> int:
        return self.recording_resolution[0]

    @property
    def vertical_resolution(self) -> int:
        return self.recording_resolution[1]

    @property
    def fps(self) -> float:
        return self.manual_fps or self._video_metadata_from_file[1]

    @property
    def frame(self) -> NDArrayUint8 | None:
        if not self.video_path:
            return None
        return self._video_metadata_from_file[2]

    @cached_property
    def _video_metadata_from_file(self) -> tuple[NDArrayInt16, float, NDArrayUint8]:
        if not self.video_path:
            msg = (
                "Requested attribute, could be: FPS, resolution, or frame could not be defined. "
                "Either define these manually (manual_fps, manual_recording_resolution), or provide path to video;"
                "frame requires video_path to be defined."
            )
            raise AttributeError(msg)

        frame, horizontal_resolution, vertical_resolution, fps = get_video_data(self.video_path)

        return np.array((horizontal_resolution, vertical_resolution), dtype=np.int16), fps, frame


class VideoMetadataMixin(_VideoMetadataBase):
    manual_video: Optional[VideoMetadata]

    @cached_property
    def video(self):
        if self.manual_video:
            return self.manual_video
        return VideoMetadata(
            manual_meters_per_pixel=self.manual_meters_per_pixel,
            manual_fps=self.manual_fps,
            manual_recording_resolution=self.manual_recording_resolution,
            metric_resolution=self.metric_resolution,
        )

    @property
    def video_metadata(self):
        return self.video.video_metadata


def video_metadata_from_object_or_metric_and_recording(
    video: Optional[VideoMetadata] = None,
    metric_resolution: Optional[float | NDArrayFp64] = None,
    recording_resolution: Optional[NDArrayFp64] = None,
    exclusive: bool = False,
) -> VideoMetadata:
    if video:
        if exclusive and metric_resolution is not None or recording_resolution is not None:
            msg = "metric_resolution and recording_resolution vs video must be defined exclusively"
            raise ValueError(msg)
        return video
    return VideoMetadata(metric_resolution=metric_resolution, manual_recording_resolution=recording_resolution)


def convert_meters_to_pixels(data: NDArrayFp64, video: VideoMetadata) -> NDArrayFp64:
    return data * video.pixels_per_meter


def inspect_video_is_none_during_inspection(inspect_video: VideoMetadata | None):
    if inspect_video is None:
        msg = "inspect_video is required to map the result from pixels to meters; required for generating inspection figure"
        raise ValueError(msg)
