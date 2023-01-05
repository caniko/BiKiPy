from functools import cached_property

from mextractor import constants
from pydantic import DirectoryPath, FilePath

from bikipy.core.typing import TrialId
from bikipy.core.video import VideoMetadata
from bikipy.ingress.plugin.base import BasePlugin


class PluginVideo(BasePlugin):
    data_path: FilePath | DirectoryPath

    ingress_key = "video"
    code_key = "video"
    default_trial_argument_key = "manual_video"
    human_readable_index = "Video"

    @cached_property
    def video(self) -> VideoMetadata:
        if self.data_path.suffix == constants.DUMP_PATH_SUFFIX:
            return VideoMetadata.from_mextractor(self.data_path)
        return VideoMetadata.from_path(video_path=self.data_path)

    def trialwise_and_metadata(self, trial_id: TrialId, naive: bool = False) -> VideoMetadata:
        return self.video

    @property
    def globally_defined(self) -> VideoMetadata:
        return self.video
