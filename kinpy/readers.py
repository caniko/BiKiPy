from collections.abc import Iterable
import pandas as pd
import numpy as np

from .compute.midpoints import compute_from_dlc_df


class DeepLabCutReader:
    def __init__(
        self,
        df: pd.DataFrame,
        video_res: tuple,
        data_label: str = None,
        midpoint_pairs: Iterable = None,
        future_scaling: bool = False,
        min_like: float = 0.95,
        invert_y: bool = False,
    ):
        """
        :param df: Kinematic data from DeepLabCut ingested as a pd.DataFrame
        :param video_res: tuple-like
            The resolution of the videos that are being analyzed
        :param data_label: string, default None
            Label for the data
        :param midpoint_pairs: list-like, default None
            List-like structure of labels that consist of pairs that should have their
            center computed. The centered pairs will be replaced by the center set
        :param future_scaling: boolean, default False
            Scales the coordinates with respect to their min and max.
            True requires x_max and y_max
        :param min_like: float, default 0.90
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
            raise ValueError(msg)

        if invert_y and not self.y_res:
            msg = "video_res needs to be defined if the y-axis is to be inverted"
            raise ValueError(msg)

        self.df = df
        self.data_label = data_label
        self.future_scaling = future_scaling
        self.min_like = min_like

        # Get the name of regions of interest
        multi_index = list(self.df)
        regions_of_interest = []
        for i in range(0, len(multi_index), 3):
            regions_of_interest.append(multi_index[i][0])
        self.regions_of_interest = regions_of_interest

        self.invalid_points = {}
        self.valid_points = {}
        for region_of_interest in self.regions_of_interest:
            self.invalid_points[region_of_interest] = (
                self.df[(region_of_interest, "likelihood")] < self.min_like
            )
            self.valid_points[region_of_interest] = np.logical_not(
                self.invalid_points[region_of_interest]
            )

            # Remove likelihood values below min_like value
            self.df.loc[
                self.invalid_points[region_of_interest],
                [(region_of_interest, "x"), (region_of_interest, "y")],
            ] = np.nan

            # Invert y-axis
            if invert_y:
                self.df[(region_of_interest, "y")] = self.df[
                    (region_of_interest, "y")
                ].map(lambda y: self.y_res - y)

        if midpoint_pairs:
            for pair in midpoint_pairs:
                if not (
                    pair[0] in self.regions_of_interest
                    and pair[1] in self.regions_of_interest
                ):
                    msg = (
                        f"The region of interest names must be referred to with "
                        f"their names, and be string:\n"
                        f"pair: {pair}\nregions_of_interest: {self.regions_of_interest}"
                    )
                    raise ValueError(msg)

            self.df = self.add_regions_of_interest_to_df(
                master=self.df,
                new_data=compute_from_dlc_df(
                    self.df, point_pair_names=midpoint_pairs, min_likelihood=0.95
                ),
            )

    @classmethod
    def from_video(cls, video_path, *args, **kwargs):
        """Initialize class using data from a sample video file

        :param video_path: Path to the video file that was used for generating dataset in DeepLabCut
        :param args: Arguments for the class.__init__
        :param kwargs: Keyword-arguments for the class.__init__
        :return: DeepLabCutReader instance
        """
        try:
            from kinpy.video_analysis import get_video_data
        except ModuleNotFoundError:
            msg = "opencv-python is required to analyse video"
            raise ModuleNotFoundError(msg)

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
            df = pd.read_csv(
                csv_path,
                index_col=0,
                skiprows=1,
                header=[0, 1],
                na_filter=False,
                dtype={"coords": int, "x": float, "y": float, "likelihood": float},
            )
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

        df = pd.read_hdf(hdf_path)
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
        func: callable,
        dlc_df_objs: list,
        keep_labels: bool = True,
        manual_labels: tuple = None,
        kwargs_for_func: dict = None,
    ):
        """
        Method for mapping a function to a list of DeepLabCutReader objects
        
        Parameters
        ----------
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
    def add_regions_of_interest_to_df(
        master: pd.DataFrame, new_data: dict
    ) -> pd.DataFrame:
        unwrapped_data = np.vstack(
            [
                np.hstack(tuple(roi_data_set.values()))
                for roi_data_set in new_data.values()
            ]
        )
        return master.join(
            pd.DataFrame(
                unwrapped_data,
                columns=pd.MultiIndex.from_product(
                    [tuple(new_data.keys()), ["x", "y", "likelihood"]]
                ),
                index=master.index,
            )
        )
