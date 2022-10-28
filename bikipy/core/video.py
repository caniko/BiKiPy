"""
Weird dataclass hierarchy for videos, requirement by design.

Purpose of VideoMetadataMixin
Classes that define videos should have this mixin: VideoMetadata, BaseExperiment, and BaseTrial. This class is
bare metadata, and its purpose is to either initialize or relay an existing VideoMetadata object
"""
from functools import cached_property, partial
from logging import getLogger
from typing import ClassVar, Optional

import cv2
import mextractor
import numpy as np
from mextractor.extractors import extract_video
from pydantic import DirectoryPath, Field, FilePath, BaseModel
from pydantic_numpy import NDArray
from pydantic_numpy.dtype import NDArrayFp64, NDArrayInt16, NDArrayUint8

from bikipy import runtime_settings
from bikipy.core.base_class import BaseBikipy
from bikipy.core.typing import MetersPerPixel

logger = getLogger(__name__)

_TICK_END_OFFSET_RATIO = 0.9

_can_only_be_set_manually = {"meters_per_pixel", "image_resize_multiplier"}


class _VideoMetadataBase(BaseModel):
    class Config:
        keep_untouched = (cached_property,)

    meters_per_pixel: Optional[MetersPerPixel] = Field(
        description="Float or 1D array defining the meter to pixel ratio"
    )
    recording_resolution: Optional[NDArrayInt16] = Field(
        description="1D array defining the resolution of the recording"
    )
    fps: Optional[float] = Field(description="Frames per second of the recording")
    frame: Optional[FilePath | NDArrayUint8] = Field(
        description="Frame from the video stored in numpy array, use read_image_from_path to read from file paths"
    )
    minimum_frame_length: Optional[float] = Field(
        description="Must be defined in case the original frame has been resized. "
        "This might be done during bikipy ingress"
    )

    category = "video_metadata"


class VideoMetadata(_VideoMetadataBase):
    def __and__(self, other: "VideoMetadata") -> bool:
        for key in set(self.dict(exclude_unset=True)).intersection(other.dict(exclude_unset=True)):
            if np.any(self.dict(exclude_unset=True)[key] != other.dict(exclude_unset=True)[key]):
                logger.debug(
                    f"self and other are incongruent on {key}: "
                    f"{self.dict(exclude_unset=True)[key]} != {other.dict(exclude_unset=True)[key]}"
                )
                return False
        return True

    def __eq__(self, other: "VideoMetadata") -> bool:
        return set(self.dict(exclude_unset=True)) == set(other.dict(exclude_unset=True))

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
        new_metadata = slave.dict(exclude_unset=True)
        new_metadata.update(master.dict(exclude_unset=True))
        return cls(**new_metadata)

    @classmethod
    def from_path(cls, video_path: FilePath, minimum_frame_length: Optional[float] = None) -> "VideoMetadata":
        info = extract_video(path_to_video=video_path)
        return cls(
            recording_resolution=info.resolution,
            fps=info.fps,
            frame=info.image,
            minimum_frame_length=minimum_frame_length,
        )

    @classmethod
    def from_mextractor(
        cls, mextractor_dir: DirectoryPath, minimum_frame_length: Optional[float] = None
    ) -> "VideoMetadata":
        info = mextractor.load(mextractor_dir)
        return cls(
            recording_resolution=info.resolution,
            fps=info.fps,
            frame=info.image,
            minimum_frame_length=minimum_frame_length,
        )

    @cached_property
    def pixels_per_meter(self) -> MetersPerPixel | None:
        if self.meters_per_pixel is not None:
            return 1.0 / self.meters_per_pixel

    @cached_property
    def multiplied_resolution(self) -> NDArrayInt16:
        if self.image_resize_multiplier:
            return np.round(self.recording_resolution * self.image_resize_multiplier).astype(np.int16)

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

    @cached_property
    def minimum_frames_tolerance(self) -> int:
        return round(self.fps * runtime_settings.minimum_seconds_tolerance)

    @cached_property
    def maximum_frames_distraction(self) -> int:
        return round(self.fps * runtime_settings.maximum_seconds_distraction)

    @cached_property
    def image_resize_multiplier(self) -> float | None:
        if self.minimum_frame_length:
            shortest_side_size = min(self.frame.shape[:2])
            if shortest_side_size < self.minimum_frame_length:
                return self.minimum_frame_length / shortest_side_size

    @cached_property
    def greyscale_frame(self) -> NDArrayUint8:
        if len(self.frame.shape) == 3 and self.frame.shape[2] == 3:
            return cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        if len(self.frame.shape) == 2:
            return self.frame
        msg = f"The frame has an unsupported shape, {self.frame.shape}"
        raise AttributeError(msg)

    @cached_property
    def upscaled_video(self) -> "VideoMetadata":
        if not self.image_resize_multiplier:
            return self
        new_frame = cv2.resize(
            self.frame.copy(),
            (0, 0),
            fx=self.image_resize_multiplier,
            fy=self.image_resize_multiplier,
            interpolation=cv2.INTER_CUBIC,
        )
        return self.__class__(
            **self.dict(exclude={"frame", "recording_resolution"}, exclude_unset=True),
            frame=new_frame,
            recording_resolution=new_frame.shape[0:2:][::-1],
        )

    def ax_ticks_metric_to_pixel(self, ax, number_of_ticks: int = 7):
        ax.set_xticks(
            ticks=np.linspace(0.0, self.horizontal_resolution * _TICK_END_OFFSET_RATIO, number_of_ticks),
            labels=np.round(
                np.linspace(0.0, self.metric_horizontal_resolution * _TICK_END_OFFSET_RATIO, number_of_ticks),
                decimals=2,
            ),
            fontsize=self.plotting_default_font_size,
        )
        ax.set_yticks(
            ticks=np.linspace(0.0, self.vertical_resolution * _TICK_END_OFFSET_RATIO, number_of_ticks),
            labels=np.round(
                np.linspace(0.0, self.metric_vertical_resolution * _TICK_END_OFFSET_RATIO, number_of_ticks), decimals=2
            ),
            fontsize=self.plotting_default_font_size,
        )

    @cached_property
    def plotting_mean_side_length(self) -> float:
        return np.sum(self.center_pixel)

    @cached_property
    def plotting_default_font_size(self) -> float:
        return self.plotting_mean_side_length / 10.0  # sum(center) == mean

    @cached_property
    def plotting_title_font_size(self) -> float:
        return self.plotting_default_font_size * 1.25

    @cached_property
    def plotting_line_thickness(self) -> float:
        return self.plotting_default_font_size / 10.0


class VideoMetadataMixin(_VideoMetadataBase):
    manual_video: Optional[VideoMetadata] = Field(
        description="Video metadata defined from another video metadata object"
    )

    required_video_metadata_fields: ClassVar[set[str]] = set()

    @property
    def video(self) -> VideoMetadata:
        # TODO: computed_field validation
        if self.required_video_metadata_fields and (
            missing_fields := self.required_video_metadata_fields.difference(self._video.dict(exclude_unset=True))
        ):
            msg = (
                f"{self.__class__.__name__} requires {self.required_video_metadata_fields}, "
                f"but is missing {missing_fields}"
            )
            raise AttributeError(msg)
        return self._video

    @property
    def video_metadata(self):
        return self.video.dict(exclude_unset=True)

    @property
    def _video(self):
        video = VideoMetadata(
            meters_per_pixel=self.meters_per_pixel,
            fps=self.fps,
            recording_resolution=self.recording_resolution,
            frame=self.frame,
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
