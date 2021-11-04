import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial, cached_property
from typing import Any, Callable, Generator, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import DirectoryPath, BaseModel, Field, Extra

from bikipy.utils.video import get_video_data


class BaseReader(BaseModel):
    df: pd.DataFrame = Field(description="Kinematic data")
    future_scaling: bool = Field(None, description="Scales the coordinates with respect to their min and max. True requires x_max and y_max")
    pixel_resolution: Optional[Sequence[int]] = Field(None, description="The resolution of the videos that are being analyzed")
    fps: Optional[float] = None
    video_path: Optional[DirectoryPath] = None
    data_path: Optional[DirectoryPath] = None
    label: Optional[str] = None

    class Config:
        extra = Extra.allow
        keep_untouched = (cached_property,)

    def __init__(self, **data: Any):
        super().__init__(**data)

        self._region_of_interest_vs_boolean_index = None
        self._valid_point_indices = None
        self._valid_tails = None
        self._valid_slices = None
        self._validity_ratio = None

    @classmethod
    def with_video_path(cls, video_path, **data):
        _frame, x_res, y_res, fps = get_video_data(video_path)
        return cls(pixel_resolution=(x_res, y_res), fps=fps, **data)

    @property
    def region_of_interest_vs_boolean_index(self):
        return self._region_of_interest_vs_boolean_index

    @region_of_interest_vs_boolean_index.setter
    def region_of_interest_vs_boolean_index(self, value: dict):
        self._region_of_interest_vs_boolean_index = dict(value)

        self._valid_point_indices = {
            roi: np.where(self.region_of_interest_vs_boolean_index[roi])[0]
            for roi in self.regions_of_interest
        }

        self._valid_tails = {
            item: (
                self._valid_point_indices[item][0],
                self._valid_point_indices[item][-1],
            )
            for item in self.items
        }

        self._valid_slices = {
            item: slice(
                self._valid_point_indices[item][0], self._valid_point_indices[item][-1]
            )
            for item in self.items
        }

        self._validity_ratio = {
            roi: np.sum(self.region_of_interest_vs_boolean_index[roi])
            / self.df[(roi, "x")].size
            for roi in self.regions_of_interest
        }

    @property
    def valid_slices(self):
        if not self._valid_slices:
            msg = "region_of_interest_vs_boolean_index has to be defined for the definition of valid_slices"
            raise AttributeError(msg)
        return self._valid_slices

    @property
    def validity_ratio(self):
        if not self._validity_ratio:
            msg = "region_of_interest_vs_boolean_index has to be defined for the definition of validity_ratio"
            raise AttributeError(msg)
        return self._validity_ratio

    @property
    def items(self) -> tuple:
        """
        Returns
        -------
        Tuple containing the name of the columns of the DataFrame
        """
        return tuple(self.df.columns.levels[0])

    @property
    def regions_of_interest(self) -> tuple:
        return self.items

    @lru_cache
    def _find_longest_tails(self, items, as_slice: bool = True):
        left_valid_tails, right_valid_tails = np.array(
            [self._valid_tails[item] for item in items]
        ).T

        result = (left_valid_tails.max(), right_valid_tails.min())

        return slice(*result) if as_slice else result

    @classmethod
    def from_parquet(cls, data_path: Any, label: Any = None, **kwargs):
        """
        Initialize class using data from a parquet file

        Note: You should assign a value to object.label by including it as a kwarg

        :param data_path: The path to the parquet file that shall be analysed
        :param label: Label for the data
        :param kwargs: Keyword arguments for the class init-method
        :type data_path: Any
        :type label: str
        :return: BaseReader instance
        """

        return cls(
            pd.read_parquet(data_path),
            data_path=data_path,
            label=label,
            **kwargs,
        )

    @classmethod
    def init_many_mapper(
        cls,
        init_method: Callable,
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
        kwarg_loaded_init = partial(init_method, **init_kwargs)

        # Process pooling in windows is subpar and is not supported.
        if enable_process_pooling and sys.platform != "win32":
            with ProcessPoolExecutor() as executor:
                for dlc_obj in executor.map(kwarg_loaded_init, data_path, labels):
                    yield dlc_obj

        else:
            for data_path, label in zip(data_path, labels):
                yield kwarg_loaded_init(data_path, label=label)
