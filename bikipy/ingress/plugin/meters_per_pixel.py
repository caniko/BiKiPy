from functools import cached_property, lru_cache
from logging import getLogger

import numpy as np
from pydantic import DirectoryPath, FilePath, computed_field, field_validator
from pydantic_numpy.typing import NpNDArrayFp64

from bikipy.core.typing import Label
from bikipy.ingress.name_parser import PluginFileStemParser
from bikipy.ingress.plugin.core.base import BasePluginFile
from bikipy.math.geometry import meter_per_pixel_from_diagonal
from bikipy.utils.collection_utils import get_first_value_in_dict
from bikipy.utils.makesense import read_first_makesense_line

logger = getLogger(__file__)


class MeterPerPixelPluginFileStemParse(PluginFileStemParser):
    def __pop_split_till_empty__(self) -> None:
        self.annotation_method = self.split.popleft()
        self.length_meters = float(self.split.popleft())
        self.label = self.split.popleft() if self.split else None


class PluginMeterPerPixel(BasePluginFile):
    required = True

    plugin_file_stem_parser = MeterPerPixelPluginFileStemParse

    ingress_key = "meters_per_pixel"
    code_key = "meters_per_pixel"
    default_trial_argument_key = "meters_per_pixel"
    human_readable_index = "MetersPerPixel"

    @field_validator("data_path")
    def is_meter_per_pixel_file(cls, value: FilePath):
        match value.stem.split("-")[0].split(".")[-1]:
            case "meter_pixel_ratio":
                logger.warning(
                    f"The {cls.code_key} file has the meter_pixel_ratio indicating it is from an older version"
                )
            case cls.code_key:
                pass
            case _:
                msg = f"Defined file, {value.stem}, is not a {cls.code_key} file"
                raise AttributeError(msg)
        return value

    @computed_field  # type: ignore[misc]
    @cached_property
    def ratio(self) -> float:
        match self.stem_info.annotation_method:
            case "diagonal":
                return meter_per_pixel_from_diagonal(
                    *read_first_makesense_line(self.data_path), self.stem_info.length_meters
                )
            case "line":
                return self.stem_info.length_meters / np.array(read_first_makesense_line(self.data_path), dtype=float)
            case _:
                msg = f"Method {self.stem_info.annotation_method} is not supported"
                raise NotImplementedError(msg)

    def trialwise_and_metadata(self, trial_id: Label, naive: bool = False) -> float:
        self._assert_correct_scope_trialwise_metadata()
        return self.ratio

    @computed_field  # type: ignore[misc]
    @property
    def globally_defined(self) -> float:
        self._assert_correct_scope_global()
        return self.ratio


@lru_cache
def detect_meters_per_pixel_in_perimeter_directory(perimeter_dir: DirectoryPath) -> dict[str, NpNDArrayFp64]:
    return {
        (mpp := PluginMeterPerPixel(data_path=meters_per_pixel_file_path)).stem_info.label or i: mpp.ratio
        for i, meters_per_pixel_file_path in enumerate(perimeter_dir.glob("meters_per_pixel-*.csv"))
    }


@lru_cache
def first_meters_per_pixel_in_perimeter_directory(perimeter_dir: DirectoryPath):
    return get_first_value_in_dict(detect_meters_per_pixel_in_perimeter_directory(perimeter_dir))
