from functools import cached_property
from typing import Optional

import numpy as np
from pydantic import FilePath
from pydantic_numpy import NDArray

from bikipy.core.base_class import BikipyBase
from bikipy.utils.video import get_video_data


class VideoMetadataMixin(BikipyBase):
    video_path: Optional[FilePath] = None

    manual_fps: Optional[float] = None
    manual_recording_resolution: Optional[NDArray] = None
    manual_meters_per_pixel: Optional[float | NDArray] = None

    metric_resolution: Optional[NDArray] = None

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
    def recording_resolution(self) -> NDArray:
        return (
            self.manual_recording_resolution
            if self.manual_recording_resolution is not None
            else self._video_metadata[0]
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
        return self.manual_fps or self._video_metadata[1]

    @cached_property
    def _video_metadata(self) -> tuple:
        if not self.video_path:
            msg = (
                "Video metadata, FPS and resolution, must be defined. "
                "Either define these manually (manual_fps, manual_recording_resolution), or provide path to video"
            )
            raise AttributeError(msg)

        _frame, horizontal_resolution, vertical_resolution, fps = get_video_data(self.video_path)
        recording_resolution = (horizontal_resolution, vertical_resolution)

        return np.array(recording_resolution, dtype=np.int16), fps
