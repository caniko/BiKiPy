from functools import cached_property

from pydantic import DirectoryPath, FilePath, computed_field, validate_call
from pydantic_numpy import NpNDArrayFp64

from bikipy.core.typing import Label
from bikipy.ingress.plugin.core.base import BasePluginFile
from bikipy.utils.makesense import get_only_point_from_makesense


class PluginCenter(BasePluginFile):
    ingress_key = "center"
    code_key = "center"
    default_trial_argument_key = "manual_center_pixels"
    human_readable_index = "Center"

    @computed_field  # type: ignore[misc]
    @cached_property
    def only_center(self) -> NpNDArrayFp64:
        return get_only_point_from_makesense(self.data_path)

    def trialwise_and_metadata(self, trial_id: Label, naive: bool = False) -> NpNDArrayFp64:
        self._assert_correct_scope_trialwise_metadata()
        return self.only_center

    @computed_field  # type: ignore[misc]
    @property
    def globally_defined(self) -> NpNDArrayFp64:
        self._assert_correct_scope_global()
        return self.only_center
