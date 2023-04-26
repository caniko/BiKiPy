from functools import cached_property

from pydantic import DirectoryPath, FilePath, validate_arguments
from pydantic_numpy.dtype import NDArrayFp64

from bikipy.core.typing import Label
from bikipy.ingress.plugin.core.base import BasePluginFile
from bikipy.utils.makesense import get_only_point_from_makesense


class PluginCenter(BasePluginFile):
    ingress_key = "center"
    code_key = "center"
    default_trial_argument_key = "manual_center_pixels"
    human_readable_index = "Center"

    @cached_property
    def only_center(self) -> NDArrayFp64:
        return get_only_point_from_makesense(self.data_path)

    def trialwise_and_metadata(self, trial_id: Label, naive: bool = False) -> NDArrayFp64:
        self._assert_correct_scope_trialwise_metadata()
        return self.only_center

    @property
    def globally_defined(self) -> NDArrayFp64:
        self._assert_correct_scope_global()
        return self.only_center


@validate_arguments
def detect_center_in_perimeter_directory(perimeter_dir: DirectoryPath):
    return {
        _get_center_file_label(center_file_path): get_only_point_from_makesense(center_file_path)
        for center_file_path in perimeter_dir.glob("center-*.csv")
    }


@validate_arguments
def _get_center_file_label(center_file_path: FilePath):
    return center_file_path.stem.split("-")[1]
