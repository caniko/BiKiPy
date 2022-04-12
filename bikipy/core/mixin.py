from functools import cached_property
from typing import Literal, Optional

import numpy as np
from pydantic import FilePath

from bikipy.core.base_class import BikipyBase
from numpy.typing import NDArray
from bikipy.utils.video import get_video_data


class VideoMetadataMixin(BikipyBase):
    video_path: Optional[FilePath] = None
    manual_recording_resolution: Optional[NDArray] = None
    manual_fps: Optional[float] = None

    @cached_property
    def video_metadata_can_be_defined(self):
        return self.video_path or (self.manual_recording_resolution and self.manual_fps)

    @property
    def recording_resolution(self) -> np.ndarray:
        return self._video_metadata[0]

    @property
    def horizontal_resolution(self):
        return self.recording_resolution[0]

    @property
    def vertical_resolution(self):
        return self.recording_resolution[1]

    @property
    def fps(self):
        return self._video_metadata[1]

    @cached_property
    def _video_metadata(self) -> tuple:
        error_msg = "Either video_path or video metadata needs to be exclusively defined."
        if np.any(self.manual_recording_resolution) and self.manual_fps:
            if self.video_path:
                raise ValueError(error_msg)
            fps = self.manual_fps
            recording_resolution = self.manual_recording_resolution
        elif self.video_path:
            _frame, horizontal_resolution, vertical_resolution, fps = get_video_data(self.video_path)
            recording_resolution = (horizontal_resolution, vertical_resolution)
        else:
            raise ValueError(error_msg)
        return np.array(recording_resolution, dtype=np.int16), fps

    @property
    def _video_metadata_dict_manual_format(self):
        return {
            "manual_recording_resolution": self.recording_resolution,
            "manual_fps": self.fps,
        }
