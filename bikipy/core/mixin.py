from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Optional

import pandas as pd
from compress_pickle import compress_pickle
from pydantic import DirectoryPath, Field, FilePath, computed_field, validate_arguments
from pydantic_numpy.dtype import NDArrayUint8

from bikipy import runtime_settings
from bikipy.core.base import BikipyConfigModel, BikipyHashable
from bikipy.utils.plot.inspect import inspect_arg_description


class AbstractFeatureCollectorMixin(BikipyHashable, ABC):
    analysis_series_cache_directory_path: Optional[DirectoryPath]
    analysis_series_cache_file_path: Optional[Path]
    analysis_series_cache_format: str = ".lz4"

    feature_collection_cache: ClassVar[bool] = False  # TODO: Add feat

    @property
    @abstractmethod
    def _analysis_series_list(self) -> list[pd.Series]:
        ...

    @computed_field
    @cached_property
    def analysis_series_cache_path(self) -> FilePath:
        if self.analysis_series_cache_file_path:
            return self.analysis_series_cache_file_path
        if self.analysis_series_cache_directory_path:
            return self.analysis_series_cache_directory_path / f"{self.label}{self.analysis_series_cache_format}"

    @computed_field
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
    inspection_fig_output_path: Optional[Path] = Field(False, description=inspect_arg_description)
    manual_inspect_image: Optional[NDArrayUint8] = Field(
        description="Image to use as background in the plots for visualising the analysis data",
    )
    inspect_image_path: Optional[FilePath] = Field(
        description="Path to image to use as background in the plots for visualising the analysis data",
    )

    @validate_arguments
    def inspect_subdir_or_bool(self, subdir_name: str) -> DirectoryPath | bool:
        if isinstance(self.inspection_fig_output_path, Path):
            return self.inspection_fig_output_path / subdir_name
        assert isinstance(self.inspection_fig_output_path, bool)
        return self.inspection_fig_output_path

    @validate_arguments
    def save(self, manual_save_path: Optional[DirectoryPath] = None) -> None:
        if manual_save_path:
            save_directory_path = manual_save_path
        elif isinstance(self.inspection_fig_output_path, Path):
            save_directory_path = self.inspection_fig_output_path
        else:
            msg = "No path provided to save method"
            raise ValueError(msg)

        compress_pickle.dump(self, save_directory_path / "experiment.pickle.lzma")
