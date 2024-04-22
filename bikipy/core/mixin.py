import os
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Optional

import matplotlib.pyplot as plt
import pandas as pd
from compress_pickle import compress_pickle
from pydantic import DirectoryPath, Field, FilePath, computed_field

from bikipy import runtime_settings
from bikipy._constant import INSPECT_FIG_FILE_FORMAT
from bikipy.core.base import BikipyConfigModel, BikipyHashable
from bikipy.utils.misc import int_file_stem_incrementor


class AbstractFeatureCollectorMixin(BikipyHashable, ABC):
    analysis_series_cache_directory_path: Optional[DirectoryPath] = None
    analysis_series_cache_file_path: Optional[Path] = None
    analysis_series_cache_format: str = ".lz4"

    feature_collection_cache: ClassVar[bool] = False  # TODO: Add feat

    @property
    @abstractmethod
    def _analysis_series_list(self) -> list[pd.Series]: ...

    @computed_field  # type: ignore[misc]
    @cached_property
    def analysis_series_cache_path(self) -> FilePath:
        if self.analysis_series_cache_file_path:
            return self.analysis_series_cache_file_path
        if self.analysis_series_cache_directory_path:
            return self.analysis_series_cache_directory_path / f"{self.label}{self.analysis_series_cache_format}"

    @computed_field  # type: ignore[misc]
    @property
    def analysis_series(self) -> pd.Series:
        if self.analysis_series_cache_path and self.analysis_series_cache_path.exists():
            return compress_pickle.load(self.analysis_series_cache_path)

        if runtime_settings.disable_process_pooling:
            result = self.compute_analysis_series()
        else:
            try:
                result = self.compute_analysis_series()
            except Exception as e:
                msg = f"{self.category.capitalize()} ID: {self.int_id}; label: {self.label}, raised an error"
                raise AttributeError(msg) from e

        if self.analysis_series_cache_path:
            compress_pickle.dump(result, self.analysis_series_cache_path)

        self._post_feature_collection_flush()
        return result

    def compute_analysis_series(self) -> pd.Series:
        return pd.concat(self._analysis_series_list[::-1], axis=0)

    def _post_feature_collection_flush(self) -> None:
        pass


class InspectPlotMixin(BikipyConfigModel):
    inspection_fig_output_path: Optional[Path] = Field(
        None,
        description="When path to a directory it is used to define the save directory of figures that will be used for inspection",
    )

    @computed_field  # type: ignore[misc]
    @cached_property
    def is_inspecting(self) -> bool:
        return bool(self.inspection_fig_output_path)

    def save_fig(self, *subdir_branches: str, base_filename: str, fig: plt.Figure, close: bool = True) -> FilePath:
        result = self.inspection_fig_output_path / self.__class__.__name__
        for subdir_branch in subdir_branches:
            result = result / subdir_branch

        os.makedirs(result, exist_ok=True)

        save_path = int_file_stem_incrementor(result / f"0-{base_filename}{INSPECT_FIG_FILE_FORMAT}")

        fig.savefig(save_path, bbox_inches="tight")
        if close:
            plt.close(fig)
