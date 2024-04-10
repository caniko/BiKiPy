"""
Weird dataclass hierarchy for videos, requirement by design.

Purpose of VideoMetadataMixin
Classes that define videos should have this mixin: VideoMetadata, BaseExperiment, and BaseTrial. This class is
bare metadata, and its purpose is to either initialize or relay an existing VideoMetadata object
"""

from collections.abc import Iterable
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import ClassVar, Generator, Optional, Self, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mextractor.base import load
from mextractor.extractors import extract_video
from pydantic import DirectoryPath, Field, FilePath, computed_field, field_validator
from pydantic_numpy import NpNDArrayFp64
from pydantic_numpy.typing import (
    Np1DArrayBool,
    NpNDArray,
    NpNDArrayInt16,
    NpNDArrayUint8,
)

from bikipy import runtime_settings
from bikipy.core.base import BikipyModel
from bikipy.core.typing import MetersPerPixel
from bikipy.utils.image import read_image_from_path
from bikipy.utils.plot.io import ax_imshow_gray

Frame = FilePath | NpNDArrayUint8

logger = getLogger(__name__)

_TICK_END_OFFSET_RATIO = 0.9

_can_only_be_set_manually = {"meters_per_pixel", "image_resize_multiplier"}


class _VideoMetadataBase(BikipyModel):
    meters_per_pixel: Optional[MetersPerPixel] = Field(
        None, description="Float or 1D array defining the meter to pixel ratio"
    )
    manual_resolution: Optional[NpNDArrayInt16] = Field(
        None, description="1D array defining the resolution of the recording"
    )
    fps: Optional[float] = Field(None, description="Frames per second of the recording")
    duration: Optional[float] = Field(None, description="Duration of the video in seconds")
    frame: Optional[Frame] = Field(
        None, description="Frame from the video stored in numpy array, use read_image_from_path to read from file paths"
    )
    video_path: Optional[FilePath] = Field(None, description="Path to the video file")
    minimum_frame_length: Optional[int] = Field(
        600,
        description="Must be defined in case the original frame has been resized. "
        "This might be done during bikipy ingress",
    )

    category = "video_metadata"

    @field_validator("frame")
    def make_sure_frame_is_read(cls, value: Optional[Frame]) -> NpNDArrayUint8 | None:
        if value is not None:
            return read_image_from_path(value) if isinstance(value, Path) else value

    @computed_field  # type: ignore[misc]
    @cached_property
    def resolution(self) -> NpNDArrayInt16 | None:
        if self.manual_resolution is not None:
            return self.manual_resolution
        if self.frame is not None:
            return np.array([self.frame.shape[1], self.frame.shape[0]], dtype=np.int16)

    def boolean_array_to_seconds(self, boolean_array: Np1DArrayBool) -> float:
        return np.sum(boolean_array) / self.fps

    def video_read_frames(self) -> Generator[NpNDArrayUint8, None, None]:
        if not self.video_path:
            msg = (
                f"Tried to read frames of video, but the {self.__class__.__name__} "
                f"does not have a video path defined"
            )
            raise AttributeError(msg)

        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            msg = f"Error opening video file: {self.video_path}"
            raise ValueError(msg)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break

        cap.release()


class VideoMetadata(_VideoMetadataBase):
    def __and__(self, other) -> bool:
        assert isinstance(other, self.__class__)

        self_metadata = self.model_dump(exclude_unset=True)
        other_metadata = other.model_dump(exclude_unset=True)

        for key in set(self_metadata).intersection(other_metadata):
            if np.any(self_metadata[key] != other_metadata[key]):
                logger.debug(
                    f"self and other are incongruent on {key}: " f"{self_metadata[key]} != {other_metadata[key]}"
                )
                return False
        return True

    def __add__(self, other) -> Self:
        return self.join(self, other)

    @classmethod
    def join(
        cls,
        superior: Self,
        inferior: Self,
        ignore_incongruity: bool = False,
        meters_per_pixel_mean: bool = False,
    ) -> Self:
        if not (superior & inferior) and not ignore_incongruity:
            msg = "VideoMetadata are incongruent"
            raise AttributeError(msg)

        meters_per_pixel: float | None = None

        if meters_per_pixel_mean and "meters_per_pixel" in inferior and "meters_per_pixel" in superior:
            meters_per_pixel = float(np.mean([inferior.meters_per_pixel, superior.meters_per_pixel], axis=0))
        elif hasattr(superior, "meters_per_pixel") and superior.meters_per_pixel is not None:
            meters_per_pixel = superior.meters_per_pixel
        elif hasattr(inferior, "meters_per_pixel") and inferior.meters_per_pixel is not None:
            meters_per_pixel = inferior.meters_per_pixel

        return cls(
            meters_per_pixel=meters_per_pixel,
            manual_resolution=(
                superior.manual_resolution if superior.manual_resolution is not None else inferior.manual_resolution
            ),
            fps=superior.fps or inferior.fps,
            frame=superior.frame if superior.frame is not None else inferior.frame,
            video_path=superior.video_path if superior.video_path else inferior.video_path,
        )

    @classmethod
    def from_path(cls, video_path: FilePath, **kwargs) -> Self:
        info = extract_video(path_to_video=video_path)
        return cls(manual_resolution=info.resolution, fps=info.fps, frame=info.image, video_path=video_path, **kwargs)

    @classmethod
    def from_mextractor(cls, mextractor_dir: DirectoryPath, **kwargs) -> Self:
        info = load(mextractor_dir)
        return cls(manual_resolution=info.resolution, fps=info.fps, frame=info.image, **kwargs)

    @computed_field  # type: ignore[misc]
    @cached_property
    def pixels_per_meter(self) -> MetersPerPixel | None:
        if self.meters_per_pixel is not None:
            return 1.0 / self.meters_per_pixel

    @computed_field  # type: ignore[misc]
    @cached_property
    def multiplied_resolution(self) -> NpNDArrayInt16:
        if self.image_resize_multiplier == 1:
            return self.resolution
        return np.round(self.resolution * self.image_resize_multiplier).astype(np.int16)

    @computed_field  # type: ignore[misc]
    @cached_property
    def center_pixels(self) -> NpNDArrayInt16:
        return np.round(self.resolution / 2.0)

    @computed_field  # type: ignore[misc]
    @property
    def horizontal_resolution(self) -> int:
        return self.resolution[0]

    @computed_field  # type: ignore[misc]
    @property
    def vertical_resolution(self) -> int:
        return self.resolution[1]

    @computed_field  # type: ignore[misc]
    @cached_property
    def metric_resolution(self) -> NpNDArrayFp64:
        return self.resolution * self.meters_per_pixel

    @computed_field  # type: ignore[misc]
    @cached_property
    def center_meters(self) -> NpNDArrayFp64:
        return self.metric_resolution / 2.0

    @computed_field  # type: ignore[misc]
    @property
    def center_for_plot(self) -> NpNDArrayFp64:
        return self.center_pixels if self.coordinates_need_to_be_scaled_for_plot else self.center_meters

    @computed_field  # type: ignore[misc]
    @property
    def metric_horizontal_resolution(self) -> int:
        return self.metric_resolution[0]

    @computed_field  # type: ignore[misc]
    @property
    def metric_vertical_resolution(self) -> int:
        return self.metric_resolution[1]

    @computed_field  # type: ignore[misc]
    @cached_property
    def minimum_frames_tolerance(self) -> int:
        return round(self.fps * runtime_settings.minimum_seconds_tolerance)

    @computed_field  # type: ignore[misc]
    @cached_property
    def maximum_frames_distraction(self) -> int:
        return round(self.fps * runtime_settings.maximum_seconds_distraction)

    @computed_field  # type: ignore[misc]
    @cached_property
    def image_resize_multiplier(self) -> float:
        if self.minimum_frame_length and self.frame is not None:
            shortest_side_size = min(self.frame.shape[:2])
            if shortest_side_size < self.minimum_frame_length:
                return self.minimum_frame_length / shortest_side_size
        return 1.0

    @computed_field  # type: ignore[misc]
    @cached_property
    def greyscale_frame(self) -> NpNDArrayUint8 | None:
        if self.frame is None:
            return

        if len(self.frame.shape) == 3 and self.frame.shape[2] == 3:
            return cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        if len(self.frame.shape) == 2:
            return self.frame

        msg = f"The frame has an unsupported shape, {self.frame.shape}"
        raise AttributeError(msg)

    @computed_field(repr=False)  # type: ignore[misc]
    @cached_property
    def upscaled_video(self) -> Self:
        if self.image_resize_multiplier == 1:
            return self
        new_frame = cv2.resize(
            self.frame,
            (0, 0),
            fx=self.image_resize_multiplier,
            fy=self.image_resize_multiplier,
            interpolation=cv2.INTER_CUBIC,
        )

        # TODO: Replace after computed_field exclude method added to model_dump
        metadata = self.model_dump(exclude={"frame", "resolution"}, exclude_unset=True)
        metadata["frame"] = new_frame
        metadata["manual_resolution"] = new_frame.shape[0:2:][::-1]

        return self.__class__(
            **metadata
            # **self.model_dump(exclude={"frame", "resolution"}, exclude_unset=True),
            # frame=new_frame,
            # manual_resolution=new_frame.shape[0:2:][::-1],
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def plotting_mean_side_length(self) -> float:
        return np.sum(self.center_pixels)

    @computed_field  # type: ignore[misc]
    @cached_property
    def coordinates_need_to_be_scaled_for_plot(self) -> bool:
        return self.frame is not None

    def ax_ticks_metric_to_pixel(self, ax: Axes, number_of_ticks: int = 5) -> None:
        ax.set_xticks(
            ticks=np.linspace(0, self.horizontal_resolution * _TICK_END_OFFSET_RATIO, number_of_ticks),
            labels=np.round(
                np.linspace(0, self.metric_horizontal_resolution * _TICK_END_OFFSET_RATIO, number_of_ticks),
                decimals=2,
            ),
        )
        ax.set_yticks(
            ticks=np.linspace(0, self.vertical_resolution * _TICK_END_OFFSET_RATIO, number_of_ticks),
            # Notice that we are inverting the y-axis at the label level to make the metric axes have the same direction
            labels=np.round(
                np.linspace(self.metric_vertical_resolution * _TICK_END_OFFSET_RATIO, 0, number_of_ticks), decimals=2
            ),
        )

    def subplots(
        self,
        nrows: int = 1,
        ncols: int = 1,
        exclude_imaging_from_rc_coord: Optional[tuple[tuple[int, int], ...]] = None,
        **kwargs,
    ) -> tuple[Figure, Sequence[Axes]]:
        # TODO: Minimum dpi is set to 300
        # kwargs["dpi"] = max(MINIMUM_FIG_DPI, kwargs["dpi"]) if "dpi" in kwargs else MINIMUM_FIG_DPI

        fig, axes = plt.subplots(
            nrows,
            ncols,
            sharey=True,
            figsize=(
                self.upscaled_video.horizontal_resolution * ncols / 100,
                self.upscaled_video.vertical_resolution * nrows / 100,
            ),
            dpi=runtime_settings.matplotlib_dpi,
            **kwargs,
        )

        if self.frame is None:
            logger.debug("Video object was used to make subplot, but no frame was defined. Figure got no background.")
            return fig, axes

        if not isinstance(axes, Iterable):
            axes = [axes]

        if exclude_imaging_from_rc_coord:
            idx_to_exclude = [c + r * nrows for c, r in exclude_imaging_from_rc_coord]

        ax: Axes
        for idx, ax in enumerate(np.array(axes).flatten()):
            if exclude_imaging_from_rc_coord and idx in idx_to_exclude:
                ax.axis("off")
                continue

            ax_imshow_gray(ax, self.upscaled_video.greyscale_frame)
            self.ax_ticks_metric_to_pixel(ax)

        return fig, axes

    def subplot(self, **kwargs) -> tuple[Figure, Axes]:
        fig, ax = self.subplots(nrows=1, ncols=1, **kwargs)
        ax = ax[0]
        return fig, ax

    def prepare_coordinates_for_plotting(
        self, data: NpNDArray | float, manual_coordinates_as_pixels: bool = False, with_resize: bool = True
    ) -> NpNDArrayFp64 | float:
        if manual_coordinates_as_pixels or self.coordinates_need_to_be_scaled_for_plot:
            result = data * self.pixels_per_meter
            if with_resize:
                result *= self.image_resize_multiplier
            return result
        return data

    def flush(self) -> None:
        if self.frame is not None:
            self.frame = None
            del self.upscaled_video, self.greyscale_frame


VideoMetadata.model_rebuild()


class VideoMetadataMixin(_VideoMetadataBase):
    manual_video: Optional[VideoMetadata] = Field(
        None, description="Video metadata defined from another video metadata object"
    )

    required_video_metadata_fields: ClassVar[set[str]] = set()

    @computed_field(return_type=VideoMetadata)
    @property
    def video(self) -> VideoMetadata:
        video = VideoMetadata(
            meters_per_pixel=self.meters_per_pixel,
            fps=self.fps,
            manual_resolution=self.resolution,
            frame=self.frame,
            video_path=self.video_path,
        )
        if self.manual_video:
            video = VideoMetadata.join(self.manual_video, video, ignore_incongruity=True)
        return video

    def video_for_computation(self) -> VideoMetadata:
        if self.required_video_metadata_fields and (
            missing_fields := self.required_video_metadata_fields.difference(self.video.model_dump(exclude_unset=True))
        ):
            if len(missing_fields) == 1 and "resolution" in missing_fields and self.video.resolution is not None:
                # Resolution is derived from either frame or manual_resolution; there is no other OR logic
                # Hence the hands-on implementation
                return self.video

            msg = (
                f"{self.__class__.__name__} requires {self.required_video_metadata_fields}, "
                f"but is missing {missing_fields}"
            )
            raise AttributeError(msg)
        return self.video
