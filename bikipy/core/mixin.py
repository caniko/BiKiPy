from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Optional

import pandas as pd
from compress_pickle import compress_pickle
from pydantic import DirectoryPath, Field, FilePath
from pydantic_numpy.dtype import NDArrayUint8

from bikipy import runtime_settings
from bikipy.core.base import BikipyConfigModel, BikipyHashable
from bikipy.utils.plot.inspect import (
    InspectArg,
    generic_inspection_finalization,
    inspect_arg_description,
)


class AbstractFeatureCollectorMixin(BikipyHashable, ABC):
    feature_collection_cache: ClassVar[bool] = False  # TODO: Add feat

    @property
    @abstractmethod
    def _analysis_series_list(self) -> list[pd.Series, ...]:
        ...

    @property
    def analysis_series(self) -> pd.Series:
        if runtime_settings.debug:
            result = self.compute_analysis_series()
        else:
            try:
                # Concatenate and reverse the order
                result = self.compute_analysis_series()
            except Exception as e:
                msg = f"{self.category.capitalize()} ID: {self.int_id}; label: {self.label}, raised an error"
                raise AttributeError(msg) from e

        self._post_feature_collection_flush()
        return result

    def compute_analysis_series(self) -> pd.Series:
        return pd.concat(self._analysis_series_list[::-1], axis=0)

    def _post_feature_collection_flush(self) -> None:
        pass


class InspectPlotMixin(BikipyConfigModel):
    inspect_arg: InspectArg = Field(False, description=inspect_arg_description)
    manual_inspect_image: Optional[NDArrayUint8] = Field(
        description="Image to use as background in the plots for visualising the analysis data",
    )
    inspect_image_path: Optional[FilePath] = Field(
        description="Path to image to use as background in the plots for visualising the analysis data",
    )

    class_inspect_directory_name: ClassVar[Optional[str]] = None

    @cached_property
    def class_inspect_arg(self) -> InspectArg:
        if isinstance(self.inspect_arg, Path):
            result = self.inspect_arg / (self.class_inspect_directory_name or self.category)
            result.mkdir(exist_ok=True)
            return result

        return self.inspect_arg

    def save(self, manual_save_path: Optional[DirectoryPath] = None) -> None:
        if manual_save_path:
            save_directory_path = manual_save_path
        elif isinstance(self.inspect_arg, Path):
            save_directory_path = self.inspect_arg
        else:
            msg = "No path provided to save method"
            raise ValueError(msg)

        compress_pickle.dump(self, save_directory_path / "experiment.pickle.lzma")

    def inspection_finalization(self, *args, **kwargs) -> None:
        generic_inspection_finalization(self.inspect_arg, *args, **kwargs)
