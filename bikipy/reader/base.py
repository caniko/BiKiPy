from functools import lru_cache
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd

from bikipy.utils.video import get_video_data


class BaseReader:
    def __init__(
        self,
        df: pd.DataFrame,
        future_scaling: bool = False,
        pixel_resolution: Union[Sequence, None] = None,
        video_path: Any = None,
        data_path: Any = None,
        data_label: Union[str, None] = None,
    ):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            Kinematic data in a pd.DataFrame
        future_scaling : boolean, default False
            Scales the coordinates with respect to their min and max.
            True requires x_max and y_max
        pixel_resolution : Sequence
             The resolution of the videos that are being analyzed
        data_label : String; optional
            Label for the data
        """

        if pixel_resolution:
            self.pixel_resolution = pixel_resolution

            self.res_horizontal, self.res_vertical = pixel_resolution
            if not (
                isinstance(self.res_horizontal, (float, int, type(None)))
                and isinstance(self.res_vertical, (float, int, type(None)))
            ):
                msg = f"x and y max are integers; not {self.res_horizontal}; {self.res_vertical}"
                raise AttributeError(msg)
        elif video_path:
            _, self.res_horizontal, self.res_vertical, self.fps = get_video_data(data_path)
            self.pixel_resolution = (self.res_horizontal, self.res_vertical)

        self.df = df
        if not isinstance(df, pd.DataFrame):
            msg = "df has to be a pandas.DataFrame"
            raise AttributeError(msg)

        self.future_scaling = future_scaling

        self.video_path = video_path
        self.data_path = data_path
        self.data_label = data_label

        self._valid_point_boolean_indices = None
        self._valid_point_indices = None
        self._valid_tails = None
        self._valid_slices = None
        self._validity_ratio = None

    @property
    def valid_point_boolean_indices(self):
        return self._valid_point_boolean_indices

    @valid_point_boolean_indices.setter
    def valid_point_boolean_indices(self, boolean_index: Sequence):
        self._valid_point_boolean_indices = np.asarray(boolean_index, dtype=bool)

        self._valid_point_indices = {
            roi: np.where(self._valid_point_boolean_indices[roi])[0]
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
            roi: np.sum(self.valid_point_boolean_indices[roi]) / self.df[
                (roi, "x")].size
            for roi in self.regions_of_interest
        }

    @property
    def valid_slices(self):
        if not self._valid_slices:
            msg = "valid_point_boolean_indices has to be defined for the definition of valid_slices"
            raise AttributeError(msg)
        return self._valid_slices

    @property
    def validity_ratio(self):
        if not self._validity_ratio:
            msg = "valid_point_boolean_indices has to be defined for the definition of validity_ratio"
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
