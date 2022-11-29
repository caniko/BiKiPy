import os
from math import floor

from pydantic import BaseSettings, Field


class BikipyRuntimeSettings(BaseSettings):
    disable_process_pooling: bool = Field(
        False,
        description="initialize each DeepLabCutReader object with multiprocessing. Useful when initialize approximately 20 or more dlc objects",
    )
    only_physical_cores: bool = False
    disable_numba: bool = False

    ignore_pre_existing_inspection_directory: bool = False

    matplotlib_scatter_alpha: float = 0.60
    matplotlib_invert_y_axis: bool = False

    minimum_seconds_tolerance: float = 1.0 / 5.0
    maximum_seconds_distraction: float = 2.0 / 3.0

    @property
    def threads_to_use(self) -> int:
        count = os.cpu_count()
        if not self.only_physical_cores:
            count = floor(count * 1.8)
        return count


runtime_settings = BikipyRuntimeSettings()
