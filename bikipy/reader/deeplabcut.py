import collections.abc as abc
import os
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import Field

from bikipy.feature.midpoint import (
    midpoint_deeplabcut_df_computation,
    recursive_midpoint,
)
from bikipy.reader.base import BaseReader

DEEPLABCUT_DF_INIT_KWARGS = {
    "index_col": 0,
    "skiprows": 1,
    "header": [0, 1],
    "na_filter": False,
    "dtype": {"coords": int, "x": float, "y": float, "likelihood": float},
}

CROPPING_PARAMETERS_BASE = {"x1": None, "x2": None, "y1": None, "y2": None}


class DeepLabCutReader(BaseReader):
    """
    Class that stores information about a given experiment conducted with DeepLabCut
    """
    min_likelihood: float = Field(
        0.80,
        description=(
            "The minimum likelihood the coordinates of the respective row. "
            "If below the values, the coords are discarded while being replaced "
            "by numpy.NaN"
        ),
    )

    _df_needs_to_be_cleaned = True

    @cached_property
    def summary_frame(self):
        result = self.df.copy()
        if self.x_axis_crop_end_point:
            for roi in self.tracked_point_labels:
                result.loc[:, (roi, "x")] = (
                    result.loc[:, (roi, "x")] + self.x_axis_crop_end_point
                )

        if self.y_add:
            for roi in self.tracked_point_labels:
                result.loc[:, (roi, "y")] = result.loc[:, (roi, "y")] + self.y_add

        if self.midpoint_groups:
            midpoint_data, midpoint_based_midpoints = {}, {}
            for name, group in self.midpoint_groups.items():
                if all(component in self.tracked_point_labels for component in group):
                    group_points = [
                        self.get_tracking_data(component_name)
                        for component_name in group
                    ]
                    midpoint_x, midpoint_y = recursive_midpoint(group_points).T
                    midpoint_data[(name, "x")] = midpoint_x
                    midpoint_data[(name, "y")] = midpoint_y
                    midpoint_data[(name, "likelihood")] = self.reduce_likelihoods(group)
                elif all(
                    component in self.tracked_and_midpoint_labels for component in group
                ):
                    midpoint_based_midpoints[name] = group
                else:
                    msg = (
                        f"Midpoint {name}, cannot be derived as its components are "
                        f"not defined in the tracked dataset nor in midpoint_groups"
                    )
                    raise ValueError(msg)

            for name, group in midpoint_based_midpoints.items():
                group_points = []
                for component_name in group:
                    try:
                        group_points.append(self.get_tracking_data(component_name))
                    except KeyError:
                        group_points.append(
                            np.array(
                                (
                                    midpoint_data[(component_name, "x")],
                                    midpoint_data[(component_name, "y")],
                                )
                            ).T
                        )
                    midpoint_x, midpoint_y = recursive_midpoint(group_points).T
                    midpoint_data[(name, "x")] = midpoint_x
                    midpoint_data[(name, "y")] = midpoint_y
                    midpoint_data[(name, "likelihood")] = self.reduce_likelihoods(group)

            midpoint_df = pd.DataFrame.from_dict(midpoint_data)
            result = pd.concat((self.raw_df, midpoint_df), axis=1)

        return result

    @property
    def df(self):
        return self.summary_frame

    @property
    def tracked_point_labels(self) -> tuple:
        return tuple(self.raw_df.columns.levels[0])

    @cached_property
    def tracked_and_midpoint_labels(self):
        return tuple(*self.tracked_point_labels, *self.midpoint_groups.keys())

    @cached_property
    def region_of_interest_vs_boolean_index(self):
        return {
            roi: self.df[(roi, "likelihood")].values >= self.min_likelihood
            for roi in self.tracked_point_labels
        }

    @property
    def frames(self):
        return self.df.shape[0]

    @cached_property
    def raw_df(self):
        if not self._df_needs_to_be_cleaned:
            return super().raw_df

        if self.df_path.suffix == ".csv":
            return pd.read_csv(self.df_path, **DEEPLABCUT_DF_INIT_KWARGS)
        elif self.df_path.suffix == ".h5":
            return pd.read_hdf(self.df_path, **DEEPLABCUT_DF_INIT_KWARGS).droplevel(0, axis=1)
        else:
            # DeepLabCut doesn't support other formats natively, assuming the df data
            # is clean
            return super().raw_df

    @classmethod
    def init_many_map(
        cls,
        *,
        init_from: str = "hdf",
        **init_many_mapper_kwargs,
    ) -> Generator:
        """
        Create many DeepLabCutReader objects using specified mapping-function

        :param init_from: Classmethod label to use for initialization
        :param init_many_mapper_kwargs: kwargs for init_many_mapper
        :type init_from: str
        :return: Objects instanced from the respective class with the provided data
        :rtype: tuple
        """

        return cls.init_many_mapper(
            ext_to_method[init_from.lower()],
            **init_many_mapper_kwargs,
        )

    @staticmethod
    def map_function(
        func: Callable,
        dlc_df_objs: dict,
        keep_labels: bool = True,
        manual_labels: Optional[Sequence] = None,
        **kwargs_for_func,
    ) -> dict:
        """Method for mapping a function to a sequence of class objects

        Parameters
        ----------
        func: Callable
            A pre-defined function that processes DeepLabCutReader objects
        dlc_df_objs: dict
            list-like of class objects to have func (a function) mapped to them
        keep_labels: bool
            If True, the function will store the returned values along with DeepLabCutReader.
            label as keys in a dictionary
        manual_labels: tuple-like; optional
            Must have length equal to number of DeepLabCutReader objects in dlc_df_objs.
            Will create a dictionary where values will be correlated based on indexed.
        kwargs_for_func
            Keyword arguments to be passed to func

        Returns
        -------
        dict: {<data label>: <class instance>...}
        """
        if not kwargs_for_func:
            kwargs_for_func = {}

        if not manual_labels:
            if keep_labels:
                if not all([dlcDF_obj.label for dlcDF_obj in dlc_df_objs]):
                    msg = (
                        "At least one of the DeepLabCutReader objects "
                        "has no label, keep label should be set to False"
                    )
                    raise ValueError(msg)

                return {
                    dlcDF_obj.label: func(dlcDF_obj.df, **kwargs_for_func)
                    for dlcDF_obj in dlc_df_objs
                }
            else:
                return [func(dlcDF_obj, **kwargs_for_func) for dlcDF_obj in dlc_df_objs]
        else:
            return {
                label: func(dlcDF_obj, **kwargs_for_func)
                for label, dlcDF_obj in zip(manual_labels, dlc_df_objs)
            }

    def __getitem__(self, query):
        def isolate_coordinates(item):
            # remove likelihood column
            coordinates = np.delete(self.df[item].values, 2, 1)

            return coordinates

        if isinstance(query, str):
            if query not in self.items:
                msg = f"'{query}' is not in object DataFrame (self.df)"
                raise AttributeError(msg)
            return isolate_coordinates(query)

        elif isinstance(query, abc.Iterable):
            # common_slice = self._find_longest_tails(query)
            return [isolate_coordinates(item) for item in query]

        else:
            raise NotImplementedError(f"{type(query)} has no implementation")

    def reduce_likelihoods(self, tracked_point_labels: Sequence) -> np.ndarray:
        """
        Reduce likelihood values by multiplication; R^n to scalar

        :param tracked_point_labels: Regions of interest of which will have its
        likelihood values reduced
        :return: np.ndarray with the reduced likelihood values
        """
        return np.multiply.reduce(
            [
                self.raw_df.loc[:, [(point, "likelihood")]].values
                for point in tracked_point_labels
            ]
        )

    def get_tracking_data(self, label: str):
        """Returns an np.ndarray with the coordinates of label"""
        return self.raw_df.loc[:, [(label, "x"), (label, "y")]].values


def convert_hdf_to_parquet(data_paths, delete_hdf: bool = False):
    """
    Convert deeplabcut hdf files to parquet format, by replacing the filename suffix
    with parquet. Thereby, keeping the original path.

    :param data_paths:
    :param delete_hdf:
    :return:
    """
    data_path = Path(data_paths)
    parquet_path = data_path.with_suffix(".parquet")

    if not parquet_path.exists():
        pd.read_hdf(data_path, **DEEPLABCUT_DF_INIT_KWARGS).droplevel(
            0, axis=1
        ).to_parquet(parquet_path)

    if delete_hdf:
        os.remove(data_path)

    return parquet_path
