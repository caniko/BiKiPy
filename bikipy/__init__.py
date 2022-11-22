from pydantic import BaseSettings, Field


class BikipyRuntimeSettings(BaseSettings):
    disable_process_pooling: bool = Field(
        False,
        description="initialize each DeepLabCutReader object with multiprocessing. Useful when initialize approximately 20 or more dlc objects",
    )
    disable_numba: bool = False

    ignore_pre_existing_inspection_directory: bool = False

    matplotlib_scatter_alpha: float = 0.60
    matplotlib_invert_y_axis: bool = False

    minimum_seconds_tolerance: float = 1.0 / 3.0
    maximum_seconds_distraction: float = 2.0 / 3.0


runtime_settings = BikipyRuntimeSettings()
