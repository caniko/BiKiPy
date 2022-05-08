from functools import cached_property
from typing import Optional

import numpy as np
from pydantic import FilePath
from pydantic_numpy import NDArray

from bikipy.core.base_class import BikipyBase
from bikipy.utils.video import get_video_data


class VideoMetadataMixin(BikipyBase):
    video_path: Optional[FilePath] = None
    manual_recording_resolution: Optional[NDArray] = None
    manual_fps: Optional[float] = None

    @cached_property
    def video_metadata_can_be_defined(self):
        return self.video_path or (self.manual_recording_resolution and self.manual_fps)

    @property
    def recording_resolution(self) -> NDArray:
        return self.manual_recording_resolution or self._video_metadata[0]

    @property
    def horizontal_resolution(self):
        return self.recording_resolution[0]

    @property
    def vertical_resolution(self):
        return self.recording_resolution[1]

    @property
    def fps(self):
        return self.manual_fps or self._video_metadata[1]

    @cached_property
    def _video_metadata(self) -> tuple:
        if self.video_path:
            msg = "Either video_path or video metadata needs to be exclusively defined."
            raise AttributeError(msg)

        _frame, horizontal_resolution, vertical_resolution, fps = get_video_data(self.video_path)
        recording_resolution = (horizontal_resolution, vertical_resolution)

        return np.array(recording_resolution, dtype=np.int16), fps

    @property
    def _video_metadata_dict_manual_format(self):
        return {
            "manual_recording_resolution": self.recording_resolution,
            "manual_fps": self.fps,
        }
