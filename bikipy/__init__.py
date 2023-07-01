import logging
from math import floor
from typing import Any

import matplotlib
from psutil import cpu_count
from pydantic import BaseSettings, Field
from schemantic.model.project import SchemanticProjectMixin

logger = logging.getLogger(__file__)


class BikipyRuntimeSettings(BaseSettings, SchemanticProjectMixin):
    disable_process_pooling: bool = Field(
        False,
        description="initialize each DeepLabCutReader object with multiprocessing. "
        "Useful when initialize approximately 20 or more dlc objects",
    )
    only_physical_cores: bool = False
    disable_numba: bool = False

    compressed_pickle_suffix: str = ".lz4"

    ignore_pre_existing_inspection_directory: bool = False

    matplotlib_scatter_alpha: float = 0.60
    matplotlib_invert_y_axis: bool = False
    matplotlib_dpi: int = 300

    minimum_seconds_tolerance: float = 0.5
    maximum_seconds_distraction: float = 1 / 3

    testing: bool = False
    debug: bool = False

    @property
    def max_workers_in_process_pool(self) -> int:
        if self.only_physical_cores:
            return cpu_count(logical=True) - 1
        return floor(cpu_count(logical=False) * 0.7)


runtime_settings: BikipyRuntimeSettings = BikipyRuntimeSettings()
if not runtime_settings.testing:
    matplotlib.use("module://mplcairo.base")


def set_bikipy_settings_from_dict(key_value_map: dict[str, Any]) -> None:
    if not key_value_map:
        return

    global runtime_settings
    for k, v in key_value_map.items():
        setattr(runtime_settings, k, v)

    logger.info(runtime_settings)
