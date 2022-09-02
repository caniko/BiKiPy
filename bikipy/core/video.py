"""
Weird dataclass hierarchy for videos, requirement by design.

Purpose of VideoMetadataMixin
Classes that define videos should have this mixin: VideoMetadata, BaseExperiment, and BaseTrial. This class is
bare metadata, and its purpose is to either initialize or relay an existing VideoMetadata object
"""
from functools import cached_property, partial
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Optional

import cv2
import numpy as np
from mextractor.video import extract_video
from numpy import ndarray
from pydantic import Field, FilePath
from pydantic_numpy import NDArray
from pydantic_numpy.dtype import NDArrayFp64, NDArrayInt16, NDArrayUint8

from bikipy.core.base_class import BaseBikipy
from bikipy.utils.image import read_image_from_path
from bikipy.utils.video import get_video_data

logger = getLogger(__name__)

_TICK_END_OFFSET_RATIO = 0.9

_can_only_be_set_manually = {"meters_per_pixel", "image_resize_multiplier"}


class _VideoMetadataBase(BaseBikipy):
    meters_per_pixel: Optional[float | NDArrayFp64] = Field(
        description="Float or 1D array defining the meter to pixel ratio"
    )

    video_path: Optional[FilePath] = Field(
        description="Path to video, used to infer recording resolution, fps, and frame"
    )
    manual_recording_resolution: Optional[NDArrayInt16] = Field(
        description="1D array defining the resolution of the recording"
    )
    manual_fps: Optional[float] = Field(description="Frames per second of the recording")
    manual_frame: Optional[FilePath | NDArrayUint8] = Field(
        description="Frame from the video stored in numpy array, use read_image_from_path to read from file paths"
    )
    minimum_frame_length: Optional[float] = Field(
        description="Must be defined in case the original frame has been resized. "
        "This might be done during bikipy ingress"
    )


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
        cls,
        master: "VideoMetadata",
        slave: "VideoMetadata",
        ignore_incongruity: bool = False,
        meters_per_pixel_mean: bool = False,
    ) -> "VideoMetadata":
        if not (master & slave) and not ignore_incongruity:
            msg = "VideoMetadata are incongruent"
            raise AttributeError(msg)
        if meters_per_pixel_mean and "meters_per_pixel" in slave and "meters_per_pixel" in master:
            master.meters_per_pixel = np.mean([slave.meters_per_pixel, master.meters_per_pixel], axis=0)
        new_metadata = slave.manual_video_metadata
        new_metadata.update(master.manual_video_metadata)
        return cls(**new_metadata)

    @classmethod
    def with_mextractor(cls, video_path: FilePath, minimum_frame_length: Optional[float] = None):
        info = extract_video(path_to_video=video_path, compress_image=False)
        return cls(
            manual_recording_resolution=info.resolution,
            manual_fps=info.fps,
            manual_frame=info.image_array,
            minimum_frame_length=minimum_frame_length,
        )

    @cached_property
    def pixels_per_meter(self) -> float | NDArrayFp64:
        result = 1.0 / self.meters_per_pixel
        if self.image_resize_multiplier:
            result *= self.image_resize_multiplier
        return result

    @cached_property
    def recording_resolution(self) -> NDArrayInt16:
        result = (
            self.manual_recording_resolution
            if self.manual_recording_resolution is not None
            else self._video_metadata_from_file[0]
        )
        if self.image_resize_multiplier:
            result = np.round(result.astype(float) * self.image_resize_multiplier).astype(np.int16)
        return result

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

    @cached_property
    def image_resize_multiplier(self) -> float | None:
        if self.minimum_frame_length:
            shortest_side_size = min(self._raw_frame.shape[:2])
            if shortest_side_size < self.minimum_frame_length:
                return self.minimum_frame_length / shortest_side_size

    @property
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
        if self.image_resize_multiplier:
            result["image_resize_multiplier"] = self.image_resize_multiplier
        return result

    @cached_property
    def manual_video_metadata(self):
        return {
            f"manual_{key}" if key not in _can_only_be_set_manually else key: value
            for key, value in self.video_metadata.items()
        }

    @cached_property
    def frame(self) -> NDArrayUint8 | None:
        if self._raw_frame is not None:
            if not self.image_resize_multiplier:
                return self._raw_frame
            return cv2.resize(
                self._raw_frame,
                (0, 0),
                fx=self.image_resize_multiplier,
                fy=self.image_resize_multiplier,
                interpolation=cv2.INTER_CUBIC,
            )

    @cached_property
    def _raw_frame(self) -> NDArrayUint8 | None:
        if self.manual_frame is not None:
            if isinstance(self.manual_frame, (Path, str)):
                return read_image_from_path(self.manual_frame)
            return self.manual_frame
        if self.video_path:
            return self._video_metadata_from_file[2]

    @cached_property
    def _video_metadata_from_file(self) -> tuple[None, None, None] | tuple[ndarray, Any, Any]:
        if not self.video_path:
            return None, None, None

        frame, horizontal_resolution, vertical_resolution, fps = get_video_data(self.video_path)

        return np.array((horizontal_resolution, vertical_resolution), dtype=np.int16), fps, frame

    def ax_ticks_metric_to_pixel(self, ax, number_of_ticks: int = 7):
        ax.set_xticks(
            ticks=np.linspace(0.0, self.horizontal_resolution * _TICK_END_OFFSET_RATIO, number_of_ticks),
            labels=np.round(
                np.linspace(0.0, self.metric_horizontal_resolution * _TICK_END_OFFSET_RATIO, number_of_ticks),
                decimals=2,
            ),
        )
        ax.set_yticks(
            ticks=np.linspace(0.0, self.vertical_resolution * _TICK_END_OFFSET_RATIO, number_of_ticks),
            labels=np.round(
                np.linspace(0.0, self.metric_vertical_resolution * _TICK_END_OFFSET_RATIO, number_of_ticks), decimals=2
            ),
        )


class VideoMetadataMixin(_VideoMetadataBase):
    manual_video: Optional[VideoMetadata] = Field(
        description="Video metadata defined from another video metadata object"
    )

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

    @property
    def _video(self):
        video = VideoMetadata(
            video_path=self.video_path,
            meters_per_pixel=self.meters_per_pixel,
            manual_fps=self.manual_fps,
            manual_recording_resolution=self.manual_recording_resolution,
            manual_frame=self.manual_frame,
        )
        if self.manual_video:
            video = VideoMetadata.join(self.manual_video, video, ignore_incongruity=True)
        return video


def prepare_data_for_plotting(data: NDArray | float, inspect_pixels: bool, video: VideoMetadata) -> NDArrayFp64 | float:
    if inspect_pixels:
        data *= video.pixels_per_meter
    if video.image_resize_multiplier:
        data *= video.image_resize_multiplier
    return data


def inspect_video_is_none_during_inspection(inspect_video: VideoMetadata | None):
    if inspect_video is None:
        msg = (
            "inspect_video is required to map the result from pixels to meters; "
            "required for generating inspection figure"
        )
        raise ValueError(msg)


incongruity_permissive_video_join = partial(VideoMetadata.join, ignore_incongruity=True)
