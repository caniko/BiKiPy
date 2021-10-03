import collections.abc as abc
import glob
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, Union

import numpy as np
import pandas as pd

from bikipy.feature.midpoint import compute_from_dlc_df
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

    def __init__(
        self,
        *args,
        midpoint_groups: Union[Iterable, None] = None,
        min_likelihood: float = 0.80,
        x_crop_start: float = 0.0,
        y_crop_start: float = 0.0,
        invert_y: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        midpoint_groups : list-like, default None
            list-like structure of labels that consist of groups that should have their
        min_likelihood : float, default 0.90
            The minimum likelihood the coordinates of the respective row.
            If below the values, the coords are discarded while being replaced
            by numpy.NaN
        invert_y : bool, default False
            Bool if True will invert the y-axis. Useful when the user wants to work in
            traditional Cartesian coordinate system where the origin is on the bottom-left
        """

        super().__init__(*args, **kwargs)

        self.min_likelihood = float(min_likelihood)

        self.x_crop_start = float(x_crop_start)
        if self.x_crop_start:
            for roi in self.regions_of_interest:
                self.df.loc[:, (roi, "x")] = self.df.loc[:, (roi, "x")] + self.x_crop_start

        self.y_crop_start = float(y_crop_start)
        self.invert_y = invert_y

        if self.y_crop_start or self.invert_y:
            if self.invert_y and self.y_crop_start:
                y_add = self.y_crop_start - self.res_vertical
            elif self.invert_y:
                y_add = -self.res_vertical
            elif self.y_crop_start:
                y_add = self.y_crop_start

            for roi in self.regions_of_interest:
                self.df.loc[:, (roi, "y")] = self.df.loc[:, (roi, "y")] + y_add

        if midpoint_groups:
            if isinstance(midpoint_groups, dict):
                midpoint_labels = tuple(midpoint_groups.keys())
                midpoint_groups = tuple(midpoint_groups.values())
            else:
                midpoint_labels = None

            recursive_midpoint_groups, normal_groups = [], []
            for i, group in enumerate(midpoint_groups):
                if any("mid" in element for element in group):
                    recursive_midpoint_groups.append(group)
                elif any(element in self.regions_of_interest for element in group):
                    normal_groups.append(group)
                else:
                    msg = (
                        f"Index {i} in midpoint_groups:"
                        f"The region of interest names must be referred to with "
                        f"their names, and be string:\n"
                        f"group: {group}\nregions_of_interest: {self.regions_of_interest}"
                    )
                    raise ValueError(msg)

            midpoints = compute_from_dlc_df(
                self.df, point_group_names_set=normal_groups
            )

            midpoint_dict = {}
            for i, (key, data) in enumerate(midpoints.items()):
                midpoint_name = midpoint_labels[i] if midpoint_labels else key

                (
                    midpoint_dict[(midpoint_name, "x")],
                    midpoint_dict[(midpoint_name, "y")],
                ) = [
                    np.hstack(component) for component in np.hsplit(data["midpoint"], 2)
                ]
                midpoint_dict[(midpoint_name, "likelihood")] = np.hstack(
                    data["likelihood"]
                )

            self.df = self.add_regions_of_interest_to_df(
                master=self.df,
                new_data=midpoint_dict,
            )

            if recursive_midpoint_groups:
                midpoints = compute_from_dlc_df(
                    self.df, point_group_names_set=recursive_midpoint_groups
                )
                midpoint_dict = {}
                for i, (key, data) in enumerate(midpoints.items()):
                    midpoint_name = midpoint_labels[i] if midpoint_labels else key

                    (
                        midpoint_dict[(midpoint_name, "x")],
                        midpoint_dict[(midpoint_name, "y")],
                    ) = [
                        np.hstack(component)
                        for component in np.hsplit(data["midpoint"], 2)
                    ]
                    midpoint_dict[(midpoint_name, "likelihood")] = np.hstack(
                        data["likelihood"]
                    )

                self.df = self.add_regions_of_interest_to_df(
                    master=self.df,
                    new_data=midpoint_dict,
                )

        self.region_of_interest_vs_boolean_index = {
            roi: self.df[(roi, "likelihood")].values >= self.min_likelihood
            for roi in self.regions_of_interest
        }

    @property
    def frames(self):
        return self.df.shape[0]

    @classmethod
    def from_csv(cls, data_path: Any, label: Any = None, **kwargs):
        """
        Create a pd.DataFrame from a csv file in DeepLabCut (DLC) format.

        Note: You should assign a value to object.label by including it as a kwarg

        Returns
        -------
        __init__ call

        :param data_path: The path to the csv file that shall be analysed; with or without ".csv" extension
        :param label:
        :param kwargs: Keyword arguments for the class init-method
        :return:
        """

        return cls(
            pd.read_csv(data_path, **DEEPLABCUT_DF_INIT_KWARGS),
            data_path=data_path,
            label=label,
            **kwargs,
        )

    @classmethod
    def from_hdf(
        cls, data_path: Any, label: Any = None, drop_level: bool = True, **kwargs
    ):
        """
        Initialize class using data from a hdf file

        Note: You should assign a value to object.label by including it as a kwarg

        :param data_path: The path to the hdf file that shall be analysed
        :param label: Label for the data
        :param drop_level: If True, remove a potentially redundant level in DataFrame
        :param kwargs: Keyword arguments for the class init-method
        :type data_path: Any
        :type drop_level: bool
        :type label: str
        :return: DeepLabCutReader instance
        """

        df = pd.read_hdf(data_path, **DEEPLABCUT_DF_INIT_KWARGS)
        if drop_level:
            df = df.droplevel(0, axis=1)

        return cls(df, data_path=data_path, label=label, **kwargs)

    @classmethod
    def init_many(
        cls,
        *init_many_mapper_args,
        init_from: str = "hdf",
        **init_many_mapper_kwargs,
    ) -> tuple:
        """
        Create many DeepLabCutReader objects using specified mapping-function

        :param init_from: Classmethod label to use for initialization
        :param init_many_mapper_kwargs: kwargs for init_many_mapper
        :type init_from: dict
        :return: Objects instanced from the respective class with the provided data
        :rtype: tuple
        """
        ext_to_method = {
            "csv": cls.from_csv,
            "parquet": cls.from_parquet,
            "h5": cls.from_hdf,
            "hdf": cls.from_hdf,
        }
        return cls.init_many_mapper(
            ext_to_method[init_from.lower()],
            *init_many_mapper_args,
            **init_many_mapper_kwargs,
        )

    @staticmethod
    def map_function(
        func: Callable,
        dlc_df_objs: dict,
        keep_labels: bool = True,
        manual_labels: Union[Sequence, None] = None,
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

    @staticmethod
    def add_regions_of_interest_to_df(
        master: pd.DataFrame, new_data: dict
    ) -> pd.DataFrame:
        return master.join(pd.DataFrame.from_dict(new_data))

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


def convert_hdf_to_parquet(data_paths, delete_hdf: bool = False):
    """
    Convert deeplabcut hdf files to parquet format, by replacing the filename suffix
    with parquet. Thereby, keeping the original path.

    Parameters
    ----------
    data_paths
    delete_hdf

    Returns
    -------

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
