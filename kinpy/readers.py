import pandas as pd
import numpy as np

from pathlib import Path

from kinpy.compute.remove_flicks import (
    find_high_velocity_events as get_hv_loc,
    get_flicks as get_flick_loc,
    invalid_relative_part_distance as irpd,
)


class KinematicData:
    def __init__(
        self,
        df,
        video_res,
        data_label=None,
        center_bp=None,
        future_scaling=False,
        min_like=0.95,
        invert_y=False,
    ):
        """
        :param df: Kinematic data to stored and analyzed in the instance
        :param video_res: tuple-like
            The resolution of the videos that are being analyzed
        :param data_label: string, default None
            Label for the data
        :param center_bp: list-like, default None
            List-like structure of labels that consist of pairs that should have their center computed.
            The centered pairs will be replaced by the center set
        :param future_scaling: boolean, default False
            Scales the coordinates with respect to their min and max.
            True requires x_max and y_max
        :param min_like: float, default 0.90
            The minimum likelihood the coordinates of the respective row.
            If below the values, the coords are discarded while being replaced
            by numpy.NaN
        :param invert_y: bool, default False
            Bool if True will invert the y-axis. Useful when the user wants to work in traditional Cartesian coordinate
            system where the origin is on the bottom-left
        """

        self.x_res, self.y_res = video_res
        if not (
            isinstance(self.x_res, (int, float, type(None)))
            and isinstance(self.y_res, (int, float, type(None)))
        ):
            msg = f"x and y max are integers; not {self.x_res}; {self.y_res}"
            raise ValueError(msg)

        if center_bp is not None and not (
            isinstance(center_bp, list) or isinstance(center_bp, tuple)
        ):
            msg = (
                "center_bp can only be defined as list or tuple;\n"
                "the most convenient data structure for body_part names"
            )
            raise ValueError(msg)

        if invert_y and self.y_res is None:
            msg = "y_res needs to be defined if y-axis is to be inverted"
            raise ValueError(msg)

        self.df = df
        self.data_label = data_label
        self.future_scaling = future_scaling
        self.min_like = min_like

        # Get name of body parts
        multi_index = list(self.df)
        body_parts = []
        for i in range(0, len(multi_index), 3):
            body_parts.append(multi_index[i][0])
        self.body_parts = body_parts

        self.invalid_points = {}
        self.valid_points = {}
        for body_part in self.body_parts:
            self.invalid_points[body_part] = (
                self.df[(body_part, "likelihood")] < self.min_like
            )
            self.valid_points[body_part] = np.logical_not(
                self.invalid_points[body_part]
            )

            # Remove likelihood values below min_like value
            self.df.loc[
                self.invalid_points[body_part], [(body_part, "x"), (body_part, "y")]
            ] = np.nan

            # Invert y-axis
            if invert_y:
                self.df[(body_part, "y")] = self.df[(body_part, "y")].map(
                    lambda y: self.y_res - y
                )

        if center_bp is not None:
            pair_names = []
            raw_data = []
            for bp_pairs in center_bp:
                if not (
                    bp_pairs[0] in self.body_parts and bp_pairs[1] in self.body_parts
                ):
                    msg = f"The body part names must be referred to with " \
                          f"their names, and be string:\n" \
                          f"bp_pairs: {bp_pairs}\n" \
                          f"body_parts: {self.body_parts}"
                    raise ValueError(msg)

                if len(bp_pairs) != 2:
                    msg = "Each pair must consist of 2 elements"
                    raise ValueError(msg)

                name = "c_%s_%s" % bp_pairs
                pair_names.append(name)

                likelihood = (
                    self.df.loc[:, [(bp_pairs[0], "likelihood")]].values
                    + self.df.loc[:, [(bp_pairs[1], "likelihood")]].values
                ) / 2
                self.invalid_points[name] = likelihood < self.min_like
                self.valid_points[name] = np.logical_not(self.invalid_points[name])

                bp_1 = self.df.loc[:, [(bp_pairs[0], "x"), (bp_pairs[0], "y")]].values
                bp_2 = self.df.loc[:, [(bp_pairs[1], "x"), (bp_pairs[1], "y")]].values

                bp_1_mag_median = np.nanmedian(
                    np.apply_along_axis(np.linalg.norm, 1, bp_1)
                )
                bp_2_mag_median = np.nanmedian(
                    np.apply_along_axis(np.linalg.norm, 1, bp_2)
                )

                if bp_1_mag_median >= bp_2_mag_median:
                    centre_point = bp_2 + ((bp_1 - bp_2) / 2)

                else:
                    centre_point = bp_1 + ((bp_2 - bp_1) / 2)
                raw_data.extend((centre_point, likelihood))

            self.body_parts.extend(pair_names)

            self.df = self.df.join(
                pd.DataFrame(
                    np.hstack(raw_data),
                    columns=pd.MultiIndex.from_product(
                        [pair_names, ["x", "y", "likelihood"]]
                    ),
                    index=self.df.index,
                )
            )

    def __repr__(self):
        return (
            f"{__class__.__name__} data_label={self.data_label}\n"
            f"norm={self.future_scaling}, "
            f"x_res={self.x_res}, y_res={self.y_res}"
        )

    @classmethod
    def from_video(cls, video_path, *args, **kwargs):
        """Initialize class using data from a sample video file

        :param video_path: Path to the video file that was used for generating dataset in DeepLabCut
        :param args: Arguments for the class.__init__
        :param kwargs: Keyword-arguments for the class.__init__
        :return: KinematicData instance
        """
        try:
            from kinpy.video_analysis import handle_video_data
        except ModuleNotFoundError:
            msg = "opencv-python is required to analyse video"
            raise ModuleNotFoundError(msg)

        _frame, x_res, y_res = handle_video_data(video_path)
        kwargs["video_res"] = (x_res, y_res)

        if "csv_path" in kwargs:
            return cls.from_csv(*args, **kwargs)
        elif "hdf_path" in kwargs:
            return cls.from_hdf(*args, **kwargs)
        else:
            return cls(*args, **kwargs)

    @classmethod
    def from_csv(cls, csv_path, *args, **kwargs):
        """
        Python classmethod to import DataFrame from csv files with DeepLabCut (DLC) format

        Note: You should assign a value to object.data_label by including it as a kwarg

        Parameters
        ----------
        csv_path: str
            The path to the csv file that shall be analysed; with or without ".csv" extension
        kwargs:
            Keyword arguments are passed to the class init-method as arguments
        """
        if not csv_path.endswith(".csv"):
            msg = "The argument has to be a string with the name of a csv csv_path"
            raise ValueError(msg)

        path = Path(csv_path)
        if not path.exists():
            msg = "The defined path to the .csv file does not exist"
            raise ValueError(msg)

        # Import the csv csv_path
        type_dict = {"coords": int, "x": float, "y": float, "likelihood": float}

        df = pd.read_csv(
            csv_path,
            engine="c",
            index_col=0,
            skiprows=1,
            header=[0, 1],
            dtype=type_dict,
            na_filter=False,
        )

        return cls(df, *args, **kwargs)

    @classmethod
    def from_hdf(cls, hdf_path, *args, drop_level=True, **kwargs):
        """
        Python classmethod to import DataFrame from hdf files with DeepLabCut (DLC) format
        
        Note: You should assign a value to object.data_label by including it as a kwarg

        Parameters
        ----------
        h5_path: str
            The path to the hdf file that shall be analysed; with or without ".h5" extension
        kwargs:
            Keyword arguments are passed to the class init-method as arguments
        """

        df = pd.read_hdf(hdf_path)
        if drop_level:
            df = df.droplevel(0, axis=1)

        return cls(df, *args, **kwargs)

    @classmethod
    def init_many(cls, file_paths, init_from="csv", labels=None, init_kwargs={}):
        """
        Method to create many KinematicData objects simultaniously using specified classmethod for initialization
        """
        ext_to_method = {"csv": cls.from_csv, "h5": cls.from_hdf, "hdf": cls.from_hdf}
        try:
            init_method = ext_to_method[init_from]
        except KeyError:
            msg = "This file type has no init function implementation, currently"
            raise ValueError(msg)

        if labels is None:
            return [init_method(file_path, **init_kwargs) for file_path in file_paths]
        else:
            return [
                init_method(file_path, data_label=label, **init_kwargs)
                for file_path, label in zip(file_paths, labels)
            ]

    @staticmethod
    def map_function(
        func, dlcDF_objs, keep_labels=True, manual_labels=None, kwargs_for_func={}
    ):
        """
        Method for mapping a function to a list of KinematicData objects
        
        Parameters
        ----------
        func: function
            A pre-defined function that processes KinematicData objects
        dlcDF_objs: list
            List of KinematicData objects to have func (a function) mapped to them
        keep_labels: bool
            Boolean that if True, the function will store the returned values along with KinematicData.data_label as keys in a dictionary
        manual_labels: list
            Must have length equal to number of KinematicData objects in dlcDF_objs. Will create dictionary where values will be correlated based on indexed.
        kwargs_for_func:
            Keyword arguments to be passed to func
        """
        if manual_labels is None:
            if keep_labels:
                if any([dlcDF_obj.data_label is None for dlcDF_obj in dlcDF_objs]):
                    msg = 'At least one of KinematicData objects have no "data_label", keep label should be set to False in this case'
                    raise ValueError(msg)
                return {
                    dlcDF_obj.data_label: func(dlcDF_obj, **kwargs_for_func)
                    for dlcDF_obj in dlcDF_objs
                }
            else:
                return [func(dlcDF_obj, **kwargs_for_func) for dlcDF_obj in dlcDF_objs]
        else:
            return {
                label: func(dlcDF_obj, **kwargs_for_func)
                for label, dlcDF_obj in zip(manual_labels, dlcDF_objs)
            }

    def remove_flicks_hv(
        self, max_vel=100, range_thresh=100, flicks_hivel=False, save=False
    ):
        """
        Clean low likelihood and high velocity points from raw dataframe

        Parameters
        ----------
        max_vel: int, default 150
            The maximum velocity between two points.
            Will become automatically generated with reference to
            fps of respective video, x_res and y_res.
        range_thresh: int, default 50
        save: bool, default False
            Bool for saving/exporting the resulting dataframe to a .csv csv_path

        Returns
        -------
        new_df: pandas.DataFrame
            The cleaned df
        """
        if not isinstance(save, bool):
            msg = "The save variable has to be bool"
            raise ValueError(msg)

        # Remove flicks from a copy of df
        new_df = irpd(self.df.copy())

        for body_part in self.body_parts:
            """Clean flicks"""
            new_df.loc[
                get_flick_loc(new_df, body_part, max_vel),
                [(body_part, "x"), (body_part, "y")],
            ] = np.nan

            """Clean high velocity values"""
            new_df.loc[
                get_hv_loc(new_df, body_part, max_vel, range_thresh),
                [(body_part, "x"), (body_part, "y")],
            ] = np.nan

        if self.future_scaling:
            new_df.loc[:, (slice(None), "x")] = (
                new_df.loc[:, (slice(None), "x")] / self.x_res
            )
            new_df.loc[:, (slice(None), "y")] = (
                new_df.loc[:, (slice(None), "y")] / self.y_res
            )

        if save:
            csv_name = "cleaned_{}".format(self.csv_filename)
            new_df.to_csv(csv_name, sep="\t")

        return new_df

    def interpolate(self, method="linear", order=None, save=False):
        """Interpolate points that have NaN
        Parameters
        ----------
        method: str, default linear
        order: {int, None}, default None
        save: bool, default False
            Bool for saving/exporting the resulting dataframe to a .csv csv_path
        Returns
        -------
        new_df: pandas.DataFrame
            The interpolated df
        """
        if not isinstance(save, bool):
            msg = "The save variable has to be bool"
            raise ValueError(msg)

        new_df = self.remove_flicks_hv()

        for body_part in self.body_parts:
            for comp in ("x", "y"):
                new_df.loc[:, (body_part, comp)] = new_df.loc[
                    :, (body_part, comp)
                ].interpolate(method=method, order=order, limit_area="inside")

        if save:
            csv_name = "interpolated_{}".format(self.csv_filename)
            new_df.to_csv(csv_name, sep="\t")

        return new_df

    def coords(self, body_part=None, state="raw"):
        """Returns body part coordinate from"""

        def coord_list(body_part_df):
            coords = []
            for i in range(len(body_part_df.x)):
                coords.append((body_part_df.x[i], body_part_df.y[i]))
            return np.array(coords)

        use_df = self.get_state(state)
        if body_part == None:
            coords = {}
            for body_part in self.body_parts:
                coords[body_part] = coord_list(use_df[body_part])
        elif isinstance(body_part, str):
            coords = coord_list(body_part)
        else:
            msg = "body_part has to be string or None"
            raise ValueError(msg)

        return coords

    def view(self, state="raw"):
        """For viewing data frame in terminal; don't use in jupyter notebook"""
        use_df = self.get_state(state)
        for body_part in self.body_parts:
            print(use_df[body_part])

    def get_state(self, state, **kwargs):
        if state == "raw":
            return self.df
        elif state == "cleaned":
            return self.remove_flicks_hv(**kwargs)
        elif state == "interpolated":
            return self.interpolate(**kwargs)
        else:
            msg = "The state can only be raw; cleaned; interpolated"
            raise ValueError(msg)
