from typing import Iterable, Callable, Union, Sequence, MutableSequence
import pandas as pd
import numpy as np

from bikipy.compute.midpoints import compute_from_dlc_df


DEEPLABCUT_DF_INIT_KWARGS = {
    "index_col": 0,
    "skiprows": 1,
    "header": [0, 1],
    "na_filter": False,
    "dtype": {"coords": int, "x": float, "y": float, "likelihood": float},
}


class DeepLabCutReader:
    def __init__(
        self,
        df: pd.DataFrame,
        video_res: tuple,
        data_label: Union[str, None] = None,
        midpoint_groups: Union[Iterable, None] = None,
        future_scaling: bool = False,
        min_likelihood: float = 0.80,
        invert_y: bool = False,
    ):
        """ Class that stores information about a given experiment conducted with DeepLabCut

        Parameters
        ----------
        df : pandas.DataFrame
            Kinematic data from DeepLabCut ingested as a pd.DataFrame
        video_res : Sequence
             The resolution of the videos that are being analyzed
        data_label : string, optional
            Label for the data
        midpoint_groups : list-like, default None
            List-like structure of labels that consist of groups that should have their
        future_scaling : boolean, default False
            Scales the coordinates with respect to their min and max.
            True requires x_max and y_max
        min_likelihood : float, default 0.90
            The minimum likelihood the coordinates of the respective row.
            If below the values, the coords are discarded while being replaced
            by numpy.NaN
        invert_y : bool, default False
            Bool if True will invert the y-axis. Useful when the user wants to work in
            traditional Cartesian coordinate system where the origin is on the bottom-left
        """

        self.horizontal_res, self.vertical_res = video_res
        if not (
            isinstance(self.horizontal_res, (int, float, type(None)))
            and isinstance(self.vertical_res, (int, float, type(None)))
        ):
            msg = f"x and y max are integers; not {self.horizontal_res}; {self.vertical_res}"
            raise AttributeError(msg)

        self.df = df
        if not isinstance(df, pd.DataFrame):
            msg = "df has to be a pandas.DataFrame"
            raise AttributeError(msg)

        self.data_label = data_label
        self.future_scaling = future_scaling
        self.min_likelihood = min_likelihood

        if invert_y:
            for roi in self.regions_of_interest:
                self.df[(roi, "y")] = self.df[(roi, "y")].map(
                    lambda y: self.vertical_res - y
                )

        if midpoint_groups:
            for group in midpoint_groups:
                if not (
                    group[0] in self.regions_of_interest
                    and group[1] in self.regions_of_interest
                ):
                    msg = (
                        f"The region of interest names must be referred to with "
                        f"their names, and be string:\n"
                        f"group: {group}\nregions_of_interest: {self.regions_of_interest}"
                    )
                    raise ValueError(msg)

            midpoints = compute_from_dlc_df(
                self.df, point_group_names_set=midpoint_groups
            )
            midpoint_dict = {}
            for midpoint_name, data in midpoints.items():
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
                master=self.df, new_data=midpoint_dict,
            )

    @property
    def _valid_point_booleans(self) -> dict:
        """

       Returns
       -------
       dictionary; region of interest versus np.ndarray of booleans, True if data in the respective index is valid
       """
        return {
            roi: self.df[(roi, "likelihood")].values >= self.min_likelihood
            for roi in self.regions_of_interest
        }

    @property
    def valid_ratios(self) -> dict:
        """

        Returns
        -------
        dictionary; region of interest versus valid data number divided by size of data
        """
        valid_point_booleans = self._valid_point_booleans
        return {
            roi: np.sum(valid_point_booleans[roi]) / self.df[(roi, "x")].size
            for roi in self.regions_of_interest
        }

    @property
    def regions_of_interest(self) -> tuple:
        """

        Returns
        -------
        Tuple containing the name of the regions of interest in the DataFrame
        """
        return tuple(self.df.columns.levels[0])

    @property
    def frame_num(self):
        """

        Returns
        -------
        Number of frame in the DeepLabCut experiment, i.e. the maximum index in the DataFrame
        """
        return self.df.shape[0]

    @classmethod
    def from_video(cls, video_path, *args, **kwargs):
        """ Initialize class using data from a sample video file

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

        _frame, horizontal_res, vertical_res = get_video_data(video_path)
        kwargs["video_res"] = (horizontal_res, vertical_res)

        if "csv_path" in kwargs:
            return cls.from_csv(*args, **kwargs)
        elif "hdf_path" in kwargs:
            return cls.from_hdf(*args, **kwargs)
        else:
            return cls(*args, **kwargs)

    @classmethod
    def from_csv(cls, csv_path: str, *args, **kwargs):
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
    def from_hdf(cls, hdf_path: str, *args, drop_level: bool = True, **kwargs):
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
        file_paths: MutableSequence,
        init_from: str = "csv",
        labels: Union[Sequence, None] = None,
        **init_kwargs,
    ) -> list:
        """ Create many DeepLabCutReader objects using specified mapping-function

        Parameters
        ----------
        file_paths: list
            Path to the data sources that will be used to generate class instances
        init_from: str
            Classmethod label to use for initialization
        labels: tuple-like
            Sequence of labels that will be stored as self.label in the class instance
        init_kwargs: dict
            Keyword arguments for the class init-method

        Returns
        -------
        List of class objects instantiated with the use of provided data
        """
        ext_to_method = {"csv": cls.from_csv, "h5": cls.from_hdf, "hdf": cls.from_hdf}
        try:
            init_method = ext_to_method[init_from]
        except KeyError:
            msg = "This file type has no init function implementation, currently"
            raise ValueError(msg)

        if not labels:
            return [init_method(file_path, **init_kwargs) for file_path in file_paths]
        else:
            return [
                init_method(file_path, data_label=label, **init_kwargs)
                for file_path, label in zip(file_paths, labels)
            ]

    @staticmethod
    def map_function(
        func: Callable,
        dlc_df_objs: Sequence,
        keep_labels: bool = True,
        manual_labels: Union[Sequence, None] = None,
        **kwargs_for_func
    ) -> dict:
        """ Method for mapping a function to a sequence of class objects

        Parameters
        ----------
        func: Callable
            A pre-defined function that processes DeepLabCutReader objects
        dlc_df_objs: tuple-like
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
        master: pd.DataFrame, new_data: dict
    ) -> pd.DataFrame:
        return master.join(pd.DataFrame.from_dict(new_data))
