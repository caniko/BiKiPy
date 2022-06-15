from abc import ABC, abstractmethod
from collections import abc
from concurrent.futures import ProcessPoolExecutor
from functools import cached_property, lru_cache, partial
from logging import getLogger
from typing import Any, Generator, Hashable, Iterable, Optional, Sequence, TypeVar

import numpy as np
import pandas as pd
from pydantic import Field, FilePath
from pydantic_numpy import NDArray

from bikipy import ENABLE_PROCESS_POOLING
from bikipy.core.base_class import BikipyBaseHashable
from bikipy.core.mixin import VideoMetadataMixin
from bikipy.core.typing import NDArrayBool
from bikipy.utils.video import get_video_data

FILE_EXTENSION_to_PANDAS_READER = {
    ".parquet": pd.read_parquet,
    ".hdf": pd.read_hdf,
    ".h5": pd.read_hdf,
}


logger = getLogger(__name__)


class BaseReader(BikipyBaseHashable, VideoMetadataMixin, ABC):
    df_path: FilePath = Field(description="Path to kinematic data, that will be " "converted to pd.DataFrame")
    timestamp_index: Optional[Sequence] = Field(
        description="Sequence of same length as df that stores the" "timestamp of each index i.e. frame."
    )
    future_scaling: bool = Field(
        None,
        description="Scales the coordinates with respect to their min and max. " "True requires x_max and y_max",
    )
    midpoint_groups: Optional[dict] = Field(description="labels that consist of groups that should have their")
    x_axis_crop_end_point: float = Field(0.0, description="")
    y_axis_crop_end_point: float = Field(0.0, description="")
    reverse_y_axis: bool = Field(
        False,
        description=(
            "if True will invert the y-axis. Useful when the user wants to work in "
            "traditional Cartesian coordinate system where the origin is on "
            "the bottom-left"
        ),
    )

    cropping_time_seconds: float = 0.0
    crop_from_end: bool = Field(
        True,
        description="Only affective if cropping_time_seconds is not 0.0. "
        "Will crop from start instead when set to False",
    )

    @abstractmethod
    def _isolate_coordinates(self, key: Iterable[Hashable] | Hashable) -> pd.DataFrame:
        ...

    @abstractmethod
    def tracked_point_labels(self) -> tuple[str, ...]:
        """
        :return: tuple storing all regions of interest that are directly tracked, no midpoints
        """
        ...

    @cached_property
    def augmented(self) -> pd.DataFrame:
        """
        :return: Tracking and augmented data stored in the same frame. The augmented data should
        include midpoints and inner interpolations.
        """
        cloned_df = self.raw_df.copy()
        if self.cropping_time_seconds:
            if self.crop_from_end:
                cloned_df = cloned_df.iloc[self.raw_frames - self.cropping_time_seconds:]
            else:
                cloned_df = cloned_df.iloc[:self.cropping_time_seconds]
        return cloned_df

    @property
    def df(self) -> pd.DataFrame:
        return self.augmented

    def __getitem__(self, query: Iterable[Hashable] | Hashable) -> pd.DataFrame:
        if not isinstance(query, str) and isinstance(query, abc.Iterable):
            return pd.merge([self._isolate_coordinates(item) for item in query], axis=1)
        else:
            if query not in self.tracked_and_midpoint_labels:
                msg = f"'{query}' is not in object DataFrame (self.summary_frame)"
                raise AttributeError(msg)
            return self._isolate_coordinates(query)

    @property
    def raw_frames(self) -> int:
        return len(self.raw_df)

    @property
    def frames(self) -> int:
        return len(self.df)

    @cached_property
    def tracked_and_midpoint_labels(self) -> tuple[str, ...]:
        return tuple(*self.tracked_point_labels, *self.midpoint_groups)

    @cached_property
    def raw_df(self) -> pd.DataFrame:
        return FILE_EXTENSION_to_PANDAS_READER[self.df_path.suffix](self.df_path)

    @staticmethod
    def get_info_from_video_path(video_path) -> dict:
        _frame, x_res, y_res, fps = get_video_data(video_path)
        return {
            "recording_resolution": (x_res, y_res),
            "fps": fps,
        }

    @property
    def region_of_interest_to_boolean_index(self) -> dict[str, NDArrayBool]:
        raise NotImplementedError

    @cached_property
    def valid_point_indices(self) -> dict[str, NDArray]:
        return {roi: np.where(self.region_of_interest_to_boolean_index[roi])[0] for roi in self.tracked_point_labels}

    @cached_property
    def valid_tails(self) -> dict[str, tuple[int, int]]:
        return {
            item: (
                self.valid_point_indices[item][0],
                self.valid_point_indices[item][-1],
            )
            for item in self.tracked_point_labels
        }

    @cached_property
    def valid_slices(self) -> dict[str, slice]:
        return {
            item: slice(self.valid_point_indices[item][0], self.valid_point_indices[item][-1])
            for item in self.tracked_point_labels
        }

    @cached_property
    def validity_ratio(self) -> dict[str, float]:
        return {
            roi: np.sum(self.region_of_interest_to_boolean_index[roi]) / len(self.raw_df)
            for roi in self.tracked_point_labels
        }

    @property
    def x_add(self) -> float:
        # The only component that effects x is x_axis_crop_end_point
        return self.x_axis_crop_end_point

    @cached_property
    def y_add(self) -> float:
        add_y = 0
        if self.reverse_y_axis:
            add_y -= self.vertical_resolution
        if self.y_axis_crop_end_point:
            add_y += self.y_axis_crop_end_point
        return add_y

    @classmethod
    def init_many_mapper(
        cls,
        data_path: Iterable[Any],
        labels: Iterable[str],
        **init_kwargs,
    ) -> Generator:
        """
        Create many BaseReader instances using specified mapping-function for initialization

        :param init_method: Most often a classmethod that calls the init method after importing the data from specific
            data format
        :param data_path: Path to the data that will imported
        :param labels: labels of the data
        :param init_kwargs: Keyword arguments for the class init-method
        :type init_method: Callable
        :type data_path: Iterable[Any]
        :type labels: Iterable[str]
        :type init_kwargs: dict
        :return: Objects instanced from the respective class with the provided data
        :rtype: tuple
        """
        kwarg_loaded_init = partial(cls, **init_kwargs)

        # Process pooling in windows is subpar and is not supported.
        if ENABLE_PROCESS_POOLING:
            with ProcessPoolExecutor() as executor:
                for dlc_obj in executor.map(kwarg_loaded_init, data_path, labels):
                    yield dlc_obj
        else:
            for data_path, label in zip(data_path, labels):
                yield kwarg_loaded_init(data_path, label=label)


Reader = TypeVar("Reader", bound=BaseReader)


@lru_cache
def _find_longest_tails(valid_tails, items, as_slice: bool = True) -> slice | tuple[int, int]:
    left_valid_tails, right_valid_tails = np.array([valid_tails[item] for item in items]).T

    result = (left_valid_tails.max(), right_valid_tails.min())

    return slice(*result) if as_slice else result
