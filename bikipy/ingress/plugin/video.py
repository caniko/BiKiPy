from functools import cached_property

import cv2
from mextractor import constants
from pydantic import DirectoryPath, FilePath, computed_field

from bikipy.core.typing import Label
from bikipy.core.video import VideoMetadata
from bikipy.ingress.plugin.core.base import BasePlugin


class PluginVideo(BasePlugin):
    flip_frame: bool = False

    data_path: FilePath | DirectoryPath

    ingress_key = "video"
    code_key = "video"
    default_trial_argument_key = "manual_video"
    human_readable_index = "Video"

    @computed_field
    @cached_property
    def video(self) -> VideoMetadata:
        result = (
            VideoMetadata.from_mextractor(self.data_path)
            if self.data_path.suffix == constants.DUMP_PATH_SUFFIX
            else VideoMetadata.from_path(video_path=self.data_path)
        )
        if self.flip_frame:
            result.frame = cv2.flip(result.frame, 0)
        return result

    def trialwise_and_metadata(self, trial_id: Label, naive: bool = False) -> VideoMetadata:
        self._assert_correct_scope_trialwise_metadata()
        return self.video

    @computed_field
    @property
    def globally_defined(self) -> VideoMetadata:
        self._assert_correct_scope_global()
        return self.video
