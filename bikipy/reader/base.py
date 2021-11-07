import sys
from abc import ABC
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial, cached_property
from typing import Any, Callable, Generator, Iterable, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, Extra, FilePath, validator

from bikipy.utils.video import get_video_data


FILE_EXTENSION_VS_PANDAS_READER = {
    "parquet": pd.read_parquet,
    "hdf": pd.read_hdf,
    "h5": pd.read_hdf,
}


class BaseReader(BaseModel, ABC):
    df_path: FilePath = Field(description="Path to kinematic data, that will be "
                                          "converted to pd.DataFrame")
    future_scaling: bool = Field(
        None,
        description="Scales the coordinates with respect to their min and max. True requires x_max and y_max",
    )
    video_path: Optional[FilePath] = None
    horizontal_resolution: Optional[int] = None
    vertical_resolution: Optional[int] = None
    fps: Optional[float] = None
    midpoint_groups: Optional[dict] = Field(
        None, description="labels that consist of groups that should have their"
    )
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
    label: Optional[str] = None

    class Config:
        frozen = True
        extra = Extra.allow
        keep_untouched = (cached_property,)

    @cached_property
    def raw_df(self):
        return FILE_EXTENSION_VS_PANDAS_READER[self.df_path.suffix](self.df_path)

    @staticmethod
    def get_info_from_video_path(video_path):
        _frame, x_res, y_res, fps = get_video_data(video_path)
        return {
            "recording_resolution": (x_res, y_res),
            "fps": fps,
        }

    @validator(
        "video_path", "horizontal_resolution", "vertical_resolution", "fps", pre=True
    )
    def get_video_data(
        cls, video_path, horizontal_resolution, vertical_resolution, fps
    ):
        if video_path:
            if horizontal_resolution or vertical_resolution or fps:
                msg = "Either video_path or video metadata needs to be exclusively defined"
                raise ValueError(msg)
            _frame, horizontal_resolution, vertical_resolution, fps = get_video_data(
                video_path
            )
        return video_path, horizontal_resolution, vertical_resolution, fps

    @property
    def recording_resolution(self):
        try:
            return self.horizontal_resolution, self.vertical_resolution
        except AttributeError as e:
            msg = (
                "horizontal_resolution, vertical_resolution needs to be defined "
                "for recording_resolution to be defined"
            )
            raise AttributeError(msg) from e

    @property
    def region_of_interest_vs_boolean_index(self):
        raise NotImplementedError

    @property
    def tracked_point_labels(self) -> tuple:
        raise NotImplementedError

    @cached_property
    def valid_point_indices(self):
        return {
            roi: np.where(self.region_of_interest_vs_boolean_index[roi])[0]
            for roi in self.tracked_point_labels
        }

    @cached_property
    def valid_tails(self):
        return {
            item: (
                self.valid_point_indices[item][0],
                self.valid_point_indices[item][-1],
            )
            for item in self.tracked_point_labels
        }

    @cached_property
    def valid_slices(self):
        return {
            item: slice(
                self.valid_point_indices[item][0], self.valid_point_indices[item][-1]
            )
            for item in self.tracked_point_labels
        }

    @cached_property
    def validity_ratio(self):
        return {
            roi: np.sum(self.region_of_interest_vs_boolean_index[roi])
            / self.raw_df[(roi, "x")].size
            for roi in self.tracked_point_labels
        }

    @property
    def x_add(self):
        # The only component that effects x is x_axis_crop_end_point
        return self.x_axis_crop_end_point

    @cached_property
    def y_add(self):
        add_y = 0
        if self.reverse_y_axis:
            add_y -= self.vertical_resolution
        if self.y_axis_crop_end_point:
            add_y += self.y_axis_crop_end_point
        return add_y

    @lru_cache
    def _find_longest_tails(self, items, as_slice: bool = True):
        left_valid_tails, right_valid_tails = np.array(
            [self.valid_tails[item] for item in items]
        ).T

        result = (left_valid_tails.max(), right_valid_tails.min())

        return slice(*result) if as_slice else result

    @classmethod
    def init_many_mapper(
        cls,
        data_path: Iterable[Any],
        labels: Iterable[str],
        enable_process_pooling: bool = True,
        **init_kwargs,
    ) -> Generator:
        """
        Create many BaseReader instances using specified mapping-function for initialization

        :param init_method: Most often a classmethod that calls the init method after importing the data from specific
            data format
        :param data_path: Path to the data that will imported
        :param labels: labels of the data
        :param enable_process_pooling: If True, initialize each DeepLabCutReader object with multiprocessing.
            Useful when initialize approximately 20 or more dlc objects
        :param init_kwargs: Keyword arguments for the class init-method
        :type init_method: Callable
        :type data_path: Iterable[Any]
        :type labels: Iterable[str]
        :type enable_process_pooling: bool
        :type init_kwargs: dict
        :return: Objects instanced from the respective class with the provided data
        :rtype: tuple
        """
        kwarg_loaded_init = partial(cls, **init_kwargs)

        # Process pooling in windows is subpar and is not supported.
        if enable_process_pooling and sys.platform != "win32":
            with ProcessPoolExecutor() as executor:
                for dlc_obj in executor.map(kwarg_loaded_init, data_path, labels):
                    yield dlc_obj

        else:
            for data_path, label in zip(data_path, labels):
                yield kwarg_loaded_init(data_path, label=label)
