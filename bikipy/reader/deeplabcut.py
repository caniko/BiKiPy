from concurrent.futures import ProcessPoolExecutor
import collections.abc as abc
from functools import partial
from typing import (
    AnyStr,
    Callable,
    Dict,
    Iterable,
    List,
    Sequence,
    SupportsFloat,
    Union,
)

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
            List-like structure of labels that consist of groups that should have their
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
                self.df[(roi, "x")] = self.df[(roi, "x")].map(
                    lambda x: self.x_crop_start + x
                )

        self.y_crop_start = float(y_crop_start)
        self.invert_y = invert_y

        if self.y_crop_start or self.invert_y:
            if self.invert_y:
                if self.y_crop_start:
                    y_mod = self.vertical_res - self.y_crop_start

                def y_map_func(y):
                    return y_mod - y

            elif self.y_crop_start:

                def y_map_func(y):
                    return self.y_crop_start + y

            for roi in self.regions_of_interest:
                self.df[(roi, "y")] = self.df[(roi, "y")].map(y_map_func)

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

        self._valid_point_booleans = {
            roi: self.df[(roi, "likelihood")].values >= self.min_likelihood
            for roi in self.regions_of_interest
        }

        self._valid_point_indexes = {
            roi: np.where(self._valid_point_booleans[roi])[0]
            for roi in self.regions_of_interest
        }

        self._valid_tails = {
            item: (
                self._valid_point_indexes[item][0],
                self._valid_point_indexes[item][-1],
            )
            for item in self.items
        }

        self.valid_tails_slices = {
            item: slice(
                self._valid_point_indexes[item][0], self._valid_point_indexes[item][-1]
            )
            for item in self.items
        }

        self.valid_ratios = {
            roi: np.sum(self._valid_point_booleans[roi]) / self.df[(roi, "x")].size
            for roi in self.regions_of_interest
        }

    def __getitem__(self, query):
        def isolate_coordinates(item):
            # remove likelihood col
            coordinates = np.delete(self.df[item].values, 2, 1)
            # clean values beneath min likelihood
            coordinates[np.logical_not(self._valid_point_booleans[item])] = np.nan
            return coordinates

        if isinstance(query, str):
            if query not in self.items:
                msg = f"'{query}' is not in object DataFrame (self.df)"
                raise AttributeError(msg)
            return isolate_coordinates(query)[self.valid_tails_slices[query]]

        elif isinstance(query, abc.Sequence):
            common_slice = self.find_longest_tails(query)
            return [isolate_coordinates(item)[common_slice] for item in query]
        else:
            raise NotImplementedError(f"{type(query)} has no implementation")

    def find_longest_tails(self, items, as_slice: bool = True):
        left_valid_tails, right_valid_tails = np.array(
            [self._valid_tails[item] for item in items]
        ).T

        result = (left_valid_tails.max(), right_valid_tails.min())

        return slice(*result) if as_slice else result

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

        _frame, horizontal_res, vertical_res, _fps = get_video_data(video_path)
        kwargs["pixel_resolution"] = (horizontal_res, vertical_res)

        if "hdf_path" in kwargs:
            return cls.from_hdf(*args, **kwargs)
        elif "csv_path" in kwargs:
            return cls.from_csv(*args, **kwargs)
        else:
            return cls(*args, **kwargs)

    @classmethod
    def from_csv(cls, csv_path: AnyStr, *args, **kwargs):
        """
        Create a pd.DataFrame from a csv file in DeepLabCut (DLC) format.

        Note: You should assign a value to object.data_label by including it as a kwarg

        Parameters
        ----------
        csv_path: str
            The path to the csv file that shall be analysed; with or without ".csv" extension
        args:
            Arguments for the class init-method
        kwargs: dict
            Keyword arguments for the class init-method

        Returns
        -------
        __init__ call
        """

        try:
            df = pd.read_csv(csv_path, **DEEPLABCUT_DF_INIT_KWARGS)
        except FileNotFoundError:
            msg = f"csv_path does not exist, {csv_path}"
            raise ValueError(msg)

        return cls(df, *args, **kwargs)

    @classmethod
    def from_hdf(cls, hdf_path: AnyStr, *args, drop_level: bool = True, **kwargs):
        """
        Initialize class using data from a hdf file

        Note: You should assign a value to object.data_label by including it as a kwarg

        Parameters
        ----------
        hdf_path: str
            The path to the hdf file that shall be analysed
        args
            Arguments for the class init-method
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

        return cls(df, *args, **kwargs)

    @classmethod
    def init_many(
        cls,
        file_paths: Union[Sequence, Iterable],
        init_from: AnyStr = "hdf",
        labels: Union[Sequence, Iterable, None] = None,
        process_pooling: bool = False,
        **init_kwargs,
    ) -> List:
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
        process_pooling: bool
            If True, initialize each DeepLabCutReader object with multiprocessing.
            Useful when initialize approximately 20 or more dlc objects
        init_kwargs: dict
            Keyword arguments for the class init-method

        Returns
        -------
        List of class objects instantiated with the use of provided data
        """
        ext_to_method = {"csv": cls.from_csv, "h5": cls.from_hdf, "hdf": cls.from_hdf}
        try:
            init_method = ext_to_method[str(init_from).lower()]
        except KeyError:
            msg = "This file type has no init function implementation, currently"
            raise ValueError(msg)

        kwarg_loaded_init = partial(init_method, **init_kwargs)

        if process_pooling:
            with ProcessPoolExecutor() as executor:
                mapped = executor.map(kwarg_loaded_init, file_paths)
                dlc_objects = [result.result() for result in mapped]

            if labels:
                labels = tuple(labels)
                for i in range(len(dlc_objects)):
                    dlc_objects[i].data_label = labels[i]

        else:
            dlc_objects = [
                kwarg_loaded_init(file_path, data_label=label)
                for file_path, label in zip(file_paths, labels)
            ]

        return dlc_objects

    @staticmethod
    def map_function(
        func: Callable,
        dlc_df_objs: Dict,
        keep_labels: bool = True,
        manual_labels: Union[Sequence, None] = None,
        **kwargs_for_func,
    ) -> Dict:
        """Method for mapping a function to a sequence of class objects

        Parameters
        ----------
        func: Callable
            A pre-defined function that processes DeepLabCutReader objects
        dlc_df_objs: dict
            List-like of class objects to have func (a function) mapped to them
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
        master: pd.DataFrame, new_data: Dict
    ) -> pd.DataFrame:
        return master.join(pd.DataFrame.from_dict(new_data))
