from functools import cached_property

import cv2

from bikipy.core.typing import TrialId
from bikipy.core.video import VideoMetadata
from bikipy.ingress.plugin.base import BasePluginFile


class PluginVideo(BasePluginFile):
    ingress_key = "video_definition_strategy"
    code_key = "video"
    bikipy_trial_key = "manual_video"
    human_readable_index = "Video"

    @cached_property
    def video(self) -> VideoMetadata:
        result = VideoMetadata(video_path=self.data_path)
        if self.ingress.settings["ingress"]["frame_upscale_multiplier"] != "float":
            multiplier = float(self.ingress.settings["ingress"]["frame_upscale_multiplier"])
            result.manual_frame = cv2.resize(result.manual_frame, (0, 0), fx=multiplier, fy=multiplier)
        return result

    def trialwise_and_metadata(self, trial_id: TrialId) -> VideoMetadata:
        return self.video

    @property
    def globally_defined(self) -> VideoMetadata:
        return self.video
