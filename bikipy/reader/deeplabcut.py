import collections.abc as abc
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence, SupportsFloat, Union

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
        min_likelihood: SupportsFloat = 0.80,
        x_crop_start: SupportsFloat = 0.0,
        y_crop_start: SupportsFloat = 0.0,
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
                self.df.loc[(roi, "x")] = self.df[(roi, "x")] + self.x_crop_start

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
                self.df.loc[(roi, "y")] = self.df[(roi, "y")] + y_add

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

        self.valid_point_boolean_indices = {
            roi: self.df[(roi, "likelihood")].values >= self.min_likelihood
            for roi in self.regions_of_interest
        }

    @property
    def frames(self):
        """

        Returns
        -------
        Number of frame in the DeepLabCut experiment, i.e. the maximum index in the DataFrame
        """
        return self.df.shape[0]

    @classmethod
    def from_video(cls, video_path, *args, **kwargs):
        """
        Initialize class using data from a sample video file

        ----------
        video_path
            Path to the video file that was used for generating dataset in DeepLabCut
        args
            Arguments for the class.__init__
        kwargs
            Keyword-arguments for the class.__init__

        Returns
        -------
        __init__ call
        """
        from bikipy.utils.video import get_video_data

        _frame, res_horizontal, res_vertical, _fps = get_video_data(video_path)
        func_kwargs = {
            "pixel_resolution": (res_horizontal, res_vertical),
            "video_path": video_path,
        }

        if "hdf_path" in kwargs:
            init_func = cls.from_hdf
        elif "csv_path" in kwargs:
            init_func = cls.from_csv
        else:
            init_func = cls

        return init_func(*args, **kwargs, **func_kwargs)

    @classmethod
    def from_csv(cls, csv_path: str, data_label: Any = None, **kwargs):
        """
        Create a pd.DataFrame from a csv file in DeepLabCut (DLC) format.

        Note: You should assign a value to object.data_label by including it as a kwarg

        Parameters
        ----------
        csv_path: str
            The path to the csv file that shall be analysed; with or without ".csv" extension
        data_label : String; optional
            Label for the data
        kwargs: dict
            Keyword arguments for the class init-method

        Returns
        -------
        __init__ call
        """

        return cls(
            pd.read_csv(csv_path, **DEEPLABCUT_DF_INIT_KWARGS),
            data_path=csv_path,
            data_label=data_label,
            **kwargs,
        )

    @classmethod
    def from_hdf(
        cls, hdf_path: str, data_label: Any = None, drop_level: bool = True, **kwargs
    ):
        """
        Initialize class using data from a hdf file

        Note: You should assign a value to object.data_label by including it as a kwarg

        Parameters
        ----------
        hdf_path: str
            The path to the hdf file that shall be analysed
        data_label : String; optional
            Label for the data
        drop_level: bool
            If True, remove a potentially redundant level in DataFrame
        kwargs: dict
            Keyword arguments for the class init-method

        Returns
        -------
        __init__ call
        """

        df = pd.read_hdf(hdf_path, **DEEPLABCUT_DF_INIT_KWARGS)
        if drop_level:
            df = df.droplevel(0, axis=1)

        return cls(df, data_path=hdf_path, data_label=data_label, **kwargs)

    @classmethod
    def from_parquet(cls, hdf_path: str, data_label: Any = None, **kwargs):
        """
        Initialize class using data from a hdf file

        Note: You should assign a value to object.data_label by including it as a kwarg

        Parameters
        ----------
        hdf_path: str
            The path to the hdf file that shall be analysed
        data_label : String; optional
            Label for the data
        kwargs: dict
            Keyword arguments for the class init-method

        Returns
        -------
        __init__ call
        """

        return cls(
            pd.read_parquet(hdf_path),
            data_path=hdf_path,
            data_label=data_label,
            **kwargs,
        )

    @classmethod
    def init_many(
        cls,
        file_paths: Union[Sequence, Iterable],
        init_from: str = "hdf",
        labels: Union[Sequence, Iterable, None] = None,
        force_process_pooling: Union[bool, None] = None,
        **init_kwargs,
    ) -> list:
        """
        Create many DeepLabCutReader objects using specified mapping-function

        Parameters
        ----------
        file_paths: Iterable
            Path to the data sources that will be used to generate class instance
        init_from: str
            Classmethod label to use for initialization
        labels: tuple-like
            Sequence of labels that will be stored as self.semantic_label in the class instance
        force_process_pooling: bool
            If True, initialize each DeepLabCutReader object with multiprocessing.
            Useful when initialize approximately 20 or more dlc objects
        init_kwargs: dict
            Keyword arguments for the class init-method

        Returns
        -------
        list of class objects instantiated with the use of provided data
        """
        ext_to_method = {
            "csv": cls.from_csv,
            "parquet": cls.from_parquet,
            "h5": cls.from_hdf,
            "hdf": cls.from_hdf,
        }
        try:
            init_method = ext_to_method[str(init_from).lower()]
        except KeyError:
            msg = "This file type has no init function implementation, currently"
            raise ValueError(msg)

        kwarg_loaded_init = partial(init_method, **init_kwargs)

        # Process pooling in windows is subpar and is not supported.
        if force_process_pooling or (
            force_process_pooling is None and sys.platform != "win32"
        ):
            args = [file_paths]
            if labels:
                args.append(labels)

            with ProcessPoolExecutor() as executor:
                dlc_objects = list(executor.map(kwarg_loaded_init, *args))

        else:
            dlc_objects = [
                kwarg_loaded_init(file_path, data_label=label)
                for file_path, label in zip(file_paths, labels)
            ]

        return dlc_objects

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
            data_label as keys in a dictionary
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
                if not all([dlcDF_obj.data_label for dlcDF_obj in dlc_df_objs]):
                    msg = (
                        "At least one of the DeepLabCutReader objects "
                        "have no data_label, keep label should be set to False"
                    )
                    raise ValueError(msg)

                return {
                    dlcDF_obj.data_label: func(dlcDF_obj.df, **kwargs_for_func)
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


def convert_hdf_to_parquet(hdf_paths, delete_hdf: bool = False):
    """
    Convert deeplabcut hdf files to parquet format, by replacing the filename suffix
    with parquet. Thereby, keeping the original path.

    Parameters
    ----------
    hdf_paths
    delete_hdf

    Returns
    -------

    """
    hdf_path = Path(hdf_paths)
    parquet_path = hdf_path.with_suffix(".parquet")

    if not parquet_path.exists():
        pd.read_hdf(hdf_path, **DEEPLABCUT_DF_INIT_KWARGS).droplevel(
            0, axis=1
        ).to_parquet(parquet_path)

    if delete_hdf:
        os.remove(hdf_path)

    return parquet_path
