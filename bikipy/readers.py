from typing import Iterable, Callable, Union
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
        trim_tolerance: Union[int, None] = 2
    ):
        """
        :param df: Kinematic data from DeepLabCut ingested as a pd.DataFrame
        :param video_res: tuple-like
            The resolution of the videos that are being analyzed
        :param data_label: string, default None
            Label for the data
        :param midpoint_groups: list-like, default None
            List-like structure of labels that consist of groups that should have their
            midpoint computed.
        :param future_scaling: boolean, default False
            Scales the coordinates with respect to their min and max.
            True requires x_max and y_max
        :param min_likelihood: float, default 0.90
            The minimum likelihood the coordinates of the respective row.
            If below the values, the coords are discarded while being replaced
            by numpy.NaN
        :param invert_y: bool, default False
            Bool if True will invert the y-axis. Useful when the user wants to work in
            traditional Cartesian coordinate system where the origin is on the bottom-left
        """

        self.x_res, self.y_res = video_res
        if not (
            isinstance(self.x_res, (int, float, type(None)))
            and isinstance(self.y_res, (int, float, type(None)))
        ):
            msg = f"x and y max are integers; not {self.x_res}; {self.y_res}"
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
                self.df[(roi, "y")] = \
                    self.df[(roi, "y")].map(lambda y: self.y_res - y)

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
                midpoint_dict[(midpoint_name, "x")], midpoint_dict[(midpoint_name, "y")] = [
                    np.hstack(component) for component in np.hsplit(data["midpoint"], 2)
                ]
                midpoint_dict[(midpoint_name, "likelihood")] = np.hstack(data["likelihood"])

            self.df = self.add_regions_of_interest_to_df(
                master=self.df, new_data=midpoint_dict,
            )

    @property
    def valid_point_booleans(self):
        return {
            roi: self.df[(roi, "likelihood")].values >= self.min_likelihood
            for roi in self.regions_of_interest
        }

    @property
    def valid_ratios(self):
        valid_point_booleans = self.valid_point_booleans
        return {
            roi: np.sum(valid_point_booleans[roi]) / self.df[(roi, "x")].size
            for roi in self.regions_of_interest
        }

    @property
    def regions_of_interest(self):
        multi_indeces = list(self.df)
        return [multi_indeces[i][0] for i in range(0, len(multi_indeces), 3)]

    @property
    def frame_num(self):
        return self.df.shape[0]

    @classmethod
    def from_video(cls, video_path, *args, **kwargs):
        """Initialize class using data from a sample video file

        :param video_path: Path to the video file that was used for generating dataset in DeepLabCut
        :param args: Arguments for the class.__init__
        :param kwargs: Keyword-arguments for the class.__init__
        :return: DeepLabCutReader instance
        """
        from .video_analysis import get_video_data

        _frame, x_res, y_res = get_video_data(video_path)
        kwargs["video_res"] = (x_res, y_res)

        if "csv_path" in kwargs:
            return cls.from_csv(*args, **kwargs)
        elif "hdf_path" in kwargs:
            return cls.from_hdf(*args, **kwargs)
        else:
            return cls(*args, **kwargs)

    @classmethod
    def from_csv(cls, csv_path, *args, **kwargs):
        """Create a pd.DataFrame from a csv file in DeepLabCut (DLC) format.

        Note: You should assign a value to object.data_label by including it as a kwarg

        :param csv_path: str
            The path to the csv file that shall be analysed; with or without ".csv" extension
        :param args:
        :param kwargs: Keyword arguments are passed to the class init-method as arguments
        :return:
        """

        try:
            df = pd.read_csv(csv_path, **DEEPLABCUT_DF_INIT_KWARGS)
        except FileNotFoundError:
            msg = f"csv_path does not exist, {csv_path}"
            raise ValueError(msg)

        return cls(df, *args, **kwargs)

    @classmethod
    def from_hdf(cls, hdf_path, *args, drop_level=True, **kwargs):
        """Initialize class using data from a hdf file
        
        Note: You should assign a value to object.data_label by including it as a kwarg


        :param hdf_path: str
            The path to the hdf file that shall be analysed
        :param kwargs: dict
            Keyword arguments for the class init-method
        """

        df = pd.read_hdf(hdf_path, **DEEPLABCUT_DF_INIT_KWARGS)
        if drop_level:
            df = df.droplevel(0, axis=1)

        return cls(df, *args, **kwargs)

    @classmethod
    def init_many(
        cls,
        file_paths: list,
        init_from="csv",
        labels: tuple = None,
        init_kwargs: dict = None,
    ):
        """Create many DeepLabCutReader objects using specified mapping-function
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
        dlc_df_objs: list,
        keep_labels: bool = True,
        manual_labels: Union[tuple, None] = None,
        min_valid_ratio: Union[float, None] = 0.80,
        kwargs_for_func: Union[dict, None] = None,
    ):
        """ Method for mapping a function to a list of DeepLabCutReader objects

        :param func: function
            A pre-defined function that processes DeepLabCutReader objects
        :param dlc_df_objs: list
            List of DeepLabCutReader objects to have func (a function) mapped to them
        :param keep_labels: bool
            If True, the function will store the returned values along with
            DeepLabCutReader.data_label as keys in a dictionary
        :param manual_labels: list
            Must have length equal to number of DeepLabCutReader objects in dlc_df_objs.
            Will create a dictionary where values will be correlated based on indexed.
        :param kwargs_for_func: dict
            Keyword arguments to be passed to func
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
    def add_regions_of_interest_to_df(master: pd.DataFrame, new_data: dict) -> pd.DataFrame:
        return master.join(pd.DataFrame.from_dict(new_data))
