from functools import cached_property, lru_cache
from logging import getLogger

import numpy as np
from pydantic import DirectoryPath, FilePath, validator

from pydantic_numpy.dtype import NDArrayFp64

from bikipy.core.typing import TrialId
from bikipy.ingress.plugin.base import BasePluginFile
from bikipy.utils.collection_utils import get_first_value_in_dict
from bikipy.utils.makesense import read_first_makesense_line
from bikipy.utils.math.geometry import meter_per_pixel_from_diagonal

logger = getLogger(__file__)


class PluginMeterPerPixel(BasePluginFile):
    required = True

    ingress_key = "meters_per_pixel"
    code_key = "meters_per_pixel"
    bikipy_trial_key = "meters_per_pixel"
    human_readable_index = "MetersPerPixel"

    @validator("data_path")
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

    @cached_property
    def _info(self):
        result = super()._info

        # TODO: Onion validation pydantic v2
        assert len(result) == 4 or len(result) == 3, (
            f"The file name for {self.code_key} files consist of name, "
            f"method, and meter length delimited by a dash this file: {self.data_path.stem}"
        )

        return result

    @property
    def annotation_method(self) -> str:
        return self._info[1]

    @cached_property
    def length_meters(self) -> float:
        return float(self._info[2])

    @property
    def file_label(self) -> str | None:
        try:
            return self._info[3]
        except IndexError:
            return None

    @cached_property
    def ratio(self) -> float:
        match self.annotation_method:
            case "diagonal":
                return meter_per_pixel_from_diagonal(*read_first_makesense_line(self.data_path), self.length_meters)
            case "line":
                return self.length_meters / np.array(read_first_makesense_line(self.data_path), dtype=float)
            case _:
                msg = f"Method {self.annotation_method} is not supported"
                raise NotImplementedError(msg)

    def trialwise_and_metadata(self, trial_id: TrialId) -> float:
        return self.ratio

    @property
    def globally_defined(self) -> float:
        return self.ratio


def meters_per_pixel_file_name_to_value(file_path: FilePath, *args, **kwargs):
    return PluginMeterPerPixel(data_path=file_path).ratio


@lru_cache
def detect_meters_per_pixel_in_perimeter_directory(perimeter_dir: DirectoryPath) -> dict[str, NDArrayFp64]:
    return {
        (mpp := PluginMeterPerPixel(data_path=meters_per_pixel_file_path)).file_label or i: mpp.ratio
        for i, meters_per_pixel_file_path in enumerate(perimeter_dir.glob("meters_per_pixel-*.csv"))
    }


def first_meters_per_pixel_in_perimeter_directory(perimeter_dir: DirectoryPath):
    return get_first_value_in_dict(detect_meters_per_pixel_in_perimeter_directory(perimeter_dir))
