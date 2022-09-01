from functools import cached_property

import cv2

from bikipy.core.typing import TrialId
from bikipy.core.video import VideoMetadata
from bikipy.ingress.plugin.base import BasePluginFile


class PluginVideo(BasePluginFile):
    ingress_key = "video"
    code_key = "video"
    bikipy_trial_key = "manual_video"
    human_readable_index = "Video"

    @cached_property
    def video(self) -> VideoMetadata:
        result = VideoMetadata.with_mextractor(video_path=self.data_path)

        raw_multiplier = self.ingress.settings["ingress"]["frame_upscale_multiplier"]
        if raw_multiplier and raw_multiplier != "float":
            multiplier = float(raw_multiplier)
            result.manual_frame = cv2.resize(result.frame, (0, 0), fx=multiplier, fy=multiplier)
            result.image_resize_multiplier = multiplier
        return result

    def trialwise_and_metadata(self, trial_id: TrialId) -> VideoMetadata:
        return self.video

    @property
    def globally_defined(self) -> VideoMetadata:
        return self.video
