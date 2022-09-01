from pydantic import BaseSettings, Field


class BikipyRuntimeSettings(BaseSettings):
    disable_process_pooling: bool = Field(
        True,
        description="initialize each DeepLabCutReader object with multiprocessing. Useful when initialize approximately 20 or more dlc objects",
    )
    enable_numba: bool = True

    matplotlib_scatter_alpha: float = 0.30
    matplotlib_invert_y_axis: bool = False


runtime_settings = BikipyRuntimeSettings()
