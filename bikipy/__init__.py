from math import floor

import matplotlib
from psutil import cpu_count
from pydantic import BaseSettings, Field

matplotlib.use("Agg")


class BikipyRuntimeSettings(BaseSettings):
    disable_process_pooling: bool = Field(
        False,
        description="initialize each DeepLabCutReader object with multiprocessing. "
        "Useful when initialize approximately 20 or more dlc objects",
    )
    only_physical_cores: bool = False
    disable_numba: bool = False

    ignore_pre_existing_inspection_directory: bool = False

    matplotlib_scatter_alpha: float = 0.60
    matplotlib_invert_y_axis: bool = False

    minimum_seconds_tolerance: float = 1.0 / 5.0
    maximum_seconds_distraction: float = 2.0 / 3.0

    debug: bool = False

    @property
    def max_workers_in_process_pool(self) -> int:
        if self.only_physical_cores:
            return cpu_count(logical=True) - 1
        return floor(cpu_count(logical=False) * 0.7)


runtime_settings: BikipyRuntimeSettings = BikipyRuntimeSettings()


def set_bikipy_settings_from_dict(value: dict) -> None:
    global runtime_settings
    runtime_settings = BikipyRuntimeSettings(**value)
