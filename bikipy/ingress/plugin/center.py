from functools import cached_property

import numpy as np
from pydantic import DirectoryPath, FilePath, computed_field, validate_arguments

from bikipy.core.typing import Label
from bikipy.ingress.plugin.core.base import BasePluginFile
from bikipy.utils.makesense import get_only_point_from_makesense


class PluginCenter(BasePluginFile):
    ingress_key = "center"
    code_key = "center"
    default_trial_argument_key = "manual_center_pixels"
    human_readable_index = "Center"

    @computed_field
    @cached_property
    def only_center(self) -> np.ndarray[float, np.dtype[np.float64]]:
        return get_only_point_from_makesense(self.data_path)

    def trialwise_and_metadata(self, trial_id: Label, naive: bool = False) -> np.ndarray[float, np.dtype[np.float64]]:
        self._assert_correct_scope_trialwise_metadata()
        return self.only_center

    @computed_field
    @property
    def globally_defined(self) -> np.ndarray[float, np.dtype[np.float64]]:
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
