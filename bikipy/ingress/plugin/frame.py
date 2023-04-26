from functools import cached_property

import cv2
from pydantic import DirectoryPath, FilePath

from bikipy.core.typing import Label
from bikipy.core.video import VideoMetadata
from bikipy.ingress.plugin.base import BasePlugin


class PluginFrame(BasePlugin):
    data_path: FilePath | DirectoryPath

    ingress_key = "frame"
    code_key = "frame"
    default_trial_argument_key = "frame"
    human_readable_index = "Frame"

    @cached_property
    def frame(self) -> VideoMetadata:
        return cv2.imread(str(self.data_path))

    def trialwise_and_metadata(self, trial_id: Label, naive: bool = False) -> VideoMetadata:
        self._assert_correct_scope_trialwise_metadata()
        return self.frame

    @property
    def globally_defined(self) -> VideoMetadata:
        self._assert_correct_scope_global()
        return self.frame
