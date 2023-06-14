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
from typing import ClassVar, Generator, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mextractor.base import load
from mextractor.extractors import extract_video
from pydantic import DirectoryPath, Field, FilePath, validator
from pydantic_numpy import NDArray
from pydantic_numpy.dtype import NDArrayBool, NDArrayInt16, NDArrayUint8

from bikipy import runtime_settings
from bikipy.core.base import BikipyModel
from bikipy.core.typing import MetersPerPixel
from bikipy.utils.image import read_image_from_path
from bikipy.utils.plot.io import ax_imshow_gray

Frame = FilePath | NDArrayUint8

logger = getLogger(__name__)

_TICK_END_OFFSET_RATIO = 0.9

_can_only_be_set_manually = {"meters_per_pixel", "image_resize_multiplier"}


class _VideoMetadataBase(BikipyModel):
    meters_per_pixel: Optional[MetersPerPixel] = Field(
        description="Float or 1D array defining the meter to pixel ratio"
    )
    recording_resolution: Optional[NDArrayInt16] = Field(
        description="1D array defining the resolution of the recording"
    )
    fps: Optional[float] = Field(description="Frames per second of the recording")
    frame: Optional[Frame] = Field(
        description="Frame from the video stored in numpy array, use read_image_from_path to read from file paths"
    )
    video_path: Optional[FilePath] = Field(description="Path to the video file")
    minimum_frame_length: Optional[int] = Field(
        600,
        description="Must be defined in case the original frame has been resized. "
        "This might be done during bikipy ingress",
    )

    category = "video_metadata"

    class Config:
        keep_untouched = (cached_property,)

    @validator("frame")
    def make_sure_frame_is_read(cls, value: Frame) -> np.ndarray[int, np.dtype[np.uint8]]:
        return read_image_from_path(value) if isinstance(value, Path) else value

    @cached_property
    def resolution(self) -> np.ndarray[int, np.int16] | None:
        if self.recording_resolution is not None:
            return self.recording_resolution
        if self.frame is not None:
            return np.array([self.frame.shape[1], self.frame.shape[0]], dtype=np.int16)

    def boolean_array_to_seconds(self, boolean_array: NDArrayBool) -> float:
        return np.sum(boolean_array) / self.fps

    def video_read_frames(self) -> Generator[np.ndarray[int, np.dtype[np.uint8]]]:
        if not self.video_path:
            logger.error(
                f"Tried to read frames of video, but the {self.__class__.__name__} "
                f"does not have a video path defined"
            )
            return

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            logger.error(f"Error opening video file: {self.video_path}")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break

        cap.release()


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
        use_slave_meters_per_pixel: bool = False,
        meters_per_pixel_mean: bool = False,
    ) -> "VideoMetadata":
        if not (master & slave) and not ignore_incongruity:
            msg = "VideoMetadata are incongruent"
            raise AttributeError(msg)

        meters_per_pixel: float | None = None

        if meters_per_pixel_mean and "meters_per_pixel" in slave and "meters_per_pixel" in master:
            assert not meters_per_pixel_mean, "Multiple meters per pixel sources defined"
            meters_per_pixel = float(np.mean([slave.meters_per_pixel, master.meters_per_pixel], axis=0))
        elif use_slave_meters_per_pixel and slave.meters_per_pixel:
            assert not use_slave_meters_per_pixel, "Multiple meters per pixel sources defined"
            meters_per_pixel = slave.meters_per_pixel
        elif master.meters_per_pixel:
            meters_per_pixel = master.meters_per_pixel

        return cls(
            meters_per_pixel=meters_per_pixel,
            recording_resolution=master.recording_resolution
            if master.recording_resolution is not None
            else slave.recording_resolution,
            fps=master.fps or slave.fps,
            frame=master.frame if master.frame is not None else slave.frame,
        )

    @classmethod
    def from_path(cls, video_path: FilePath, **kwargs) -> "VideoMetadata":
        info = extract_video(path_to_video=video_path)
        return cls(recording_resolution=info.resolution, fps=info.fps, frame=info.image, **kwargs)

    @classmethod
    def from_mextractor(cls, mextractor_dir: DirectoryPath, **kwargs) -> "VideoMetadata":
        info = load(mextractor_dir)
        return cls(recording_resolution=info.resolution, fps=info.fps, frame=info.image, **kwargs)

    @cached_property
    def pixels_per_meter(self) -> MetersPerPixel | None:
        if self.meters_per_pixel is not None:
            return 1.0 / self.meters_per_pixel

    @cached_property
    def multiplied_resolution(self) -> np.ndarray[int, np.int16]:
        if self.image_resize_multiplier == 1:
            return self.recording_resolution
        return np.round(self.resolution * self.image_resize_multiplier).astype(np.int16)

    @cached_property
    def center_pixel(self) -> np.ndarray[int, np.int16]:
        return np.round(self.resolution / 2.0)

    @property
    def horizontal_resolution(self) -> int:
        return self.resolution[0]

    @property
    def vertical_resolution(self) -> int:
        return self.resolution[1]

    @cached_property
    def metric_resolution(self) -> np.ndarray[float, np.dtype[np.float64]]:
        return self.resolution * self.meters_per_pixel

    @cached_property
    def center_meters(self) -> np.ndarray[float, np.dtype[np.float64]]:
        return self.metric_resolution / 2.0

    @property
    def center_for_plot(self) -> np.ndarray[float, np.dtype[np.float64]]:
        return self.center_pixel if self.coordinates_need_to_be_scaled_for_plot else self.center_meters

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
    def image_resize_multiplier(self) -> float:
        if self.minimum_frame_length and self.frame is not None:
            shortest_side_size = min(self.frame.shape[:2])
            if shortest_side_size < self.minimum_frame_length:
                return self.minimum_frame_length / shortest_side_size
        return 1.0

    @cached_property
    def greyscale_frame(self) -> np.ndarray[int, np.dtype[np.uint8]]:
        if len(self.frame.shape) == 3 and self.frame.shape[2] == 3:
            return cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        if len(self.frame.shape) == 2:
            return self.frame
        msg = f"The frame has an unsupported shape, {self.frame.shape}"
        raise AttributeError(msg)

    @cached_property
    def upscaled_video(self) -> "VideoMetadata":
        if self.image_resize_multiplier == 1:
            return self
        new_frame = cv2.resize(
            self.frame,
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

    def ax_ticks_metric_to_pixel(self, ax: Axes, number_of_ticks: int = 7):
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
            # Notice that we are inverting the y-axis at the label level to make the metric axes have the same direction
            labels=np.round(
                np.linspace(self.metric_vertical_resolution * _TICK_END_OFFSET_RATIO, 0, number_of_ticks), decimals=2
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

    @cached_property
    def coordinates_need_to_be_scaled_for_plot(self) -> bool:
        return self.frame is not None

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
            figsize=(
                self.upscaled_video.horizontal_resolution * min(0.1, ncols / nrows),
                self.upscaled_video.vertical_resolution * min(0.1, nrows / ncols),
            ),
            **kwargs,
        )
        if self.frame is None:
            logger.debug("Video object was used to make subplot, but no frame was defined. Figure got no background.")
            return fig, axes

        if not isinstance(axes, Iterable):
            axes = [axes]

        if exclude_imaging_from_rc_coord:
            idx_to_exclude = [c + r * nrows for c, r in exclude_imaging_from_rc_coord]

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
        self, data: NDArray | float, manual_inspect_pixels: bool = False
    ) -> np.ndarray[float, np.dtype[np.float64]] | float:
        if manual_inspect_pixels or self.coordinates_need_to_be_scaled_for_plot:
            return data * self.pixels_per_meter * self.image_resize_multiplier
        return data

    def flush(self) -> None:
        if self.frame is not None:
            self.frame = None
            try:
                del self.upscaled_video
                del self.greyscale_frame
            except AttributeError as e:
                logger.debug(str(e))


class VideoMetadataMixin(_VideoMetadataBase):
    manual_video: Optional[VideoMetadata] = Field(
        description="Video metadata defined from another video metadata object"
    )

    required_video_metadata_fields: ClassVar[set[str]] = set()

    @property
    def video(self) -> VideoMetadata:
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
            recording_resolution=self.resolution,
            frame=self.frame,
        )
        if self.manual_video:
            video = VideoMetadata.join(self.manual_video, video, ignore_incongruity=True)
        return video


def inspect_video_is_none_during_inspection(inspect_video: VideoMetadata | None):
    if inspect_video is None:
        msg = (
            "inspect_video is required to map the result from pixels to meters; "
            "required for generating inspection figure"
        )
        raise ValueError(msg)
