import pandas as pd
import numpy as np

from pathlib import Path
import os

from dlca.functions.hviva import (
    get_high_velocity_locations as get_hv_loc,
    get_flicks_locations as get_flick_loc,
    invalid_relative_part_distance as irpd
)


class dlcDF:
    def __init__(
        self, df, data_label, video_path=None, center_bp=None,
        future_scaling=False, min_like=0.95, x_max=None, y_max=None
    ):
        """
        data_label: string

        video_path: string-like, default None
            Path to the video file that was used for generating dataset in DeepLabCut
        center_bp: list-like, default None
            List-like structure of labels that consist of pairs that should have their center computed.
            The centered pairs will be replaced by the center set
        future_scaling: boolean, default False
            Scales the coordinates with respect to their min and max.
            True requires x_max and y_max
        min_like: float, default 0.90
            The minimum likelihood the coordinates of the respective row.
            If below the values, the coords are discarded while being replaced
            by numpy.NaN
        x_max: {None, int}, default None
            Maximum x value, can be extracted from video sample or defined
        y_max: {None, int}, default None
            Maximum y value, can be extracted from video sample or defined
        """

        if not (isinstance(x_max, (int, float, type(None))) and
                isinstance(y_max, (int, float, type(None)))):
            msg = 'x and y max are integers; not {}; {}'.format(x_max, y_max)
            raise AttributeError(msg)

        if future_scaling is True and video_path is None and (x_max is None and y_max is None):
            msg = 'x max and y max, or vieo_file has to defined in order to future_scaling'
            raise AttributeError(msg)

        if center_bp is not None and not (isinstance(center_bp, list) or isinstance(center_bp, tuple)):
            msg = 'center_bp can only be defined as list or tuple;\nthe most convinient data structure for body_part names'
            raise AttributeError(msg)

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

        # Pair data frame with its raw source, which is a video file
        self.video_path = video_path
        if video_path is True or isinstance(video_path, str):
            try:
                from dlca.video_analysis import handle_video_data
            except ModuleNotFoundError:
                msg = 'opencv-python is required to analyse video'
                raise ModuleNotFoundError(msg)

            self.x_max, self.y_max = handle_video_data(
                video_path=video_path)[1:3]
        else:
            self.x_max = x_max
            self.y_max = y_max

        self.invalid_points = {}
        self.valid_points = {}
        for body_part in self.body_parts:
            self.invalid_points[body_part] = self.df[(body_part, 'likelihood')] < self.min_like
            self.valid_points[body_part] = np.logical_not(self.invalid_points[body_part])
            
            # Remove likelihood values below min_like value
            self.df.loc[
                self.invalid_points[body_part],
                [(body_part, 'x'), (body_part, 'y')]
            ] = np.nan

            # Invert y axis
            self.df[(body_part, 'y')] = self.df[(body_part, 'y')].map(lambda y: self.y_max - y)

        if center_bp is not None:
            pair_names = []
            raw_data = []
            for bp_pairs in center_bp:
                if not (bp_pairs[0] in self.body_parts and bp_pairs[1] in self.body_parts):
                    msg = f'The body part names must be referred to with their names, and be string:\nbp_pairs: {bp_pairs}\nbody_parts: {self.body_parts}'
                    raise AttributeError(msg)

                if len(bp_pairs) != 2:
                    msg = 'Each pair must consist of 2 elements'
                    raise AttributeError(msg)

                name = 'c_%s_%s' % bp_pairs
                pair_names.append(name)

                likelihood = (self.df.loc[:, [(bp_pairs[0], 'likelihood')]].values \
                            + self.df.loc[:, [(bp_pairs[1], 'likelihood')]].values) \
                            / 2
                self.invalid_points[name] = likelihood < self.min_like
                self.valid_points[name] = np.logical_not(self.invalid_points[name])

                bp_1 = self.df.loc[:, [(bp_pairs[0], 'x'), (bp_pairs[0], 'y')]].values
                bp_2 = self.df.loc[:, [(bp_pairs[1], 'x'), (bp_pairs[1], 'y')]].values

                bp_1_mag_median = np.nanmedian(np.apply_along_axis(np.linalg.norm, 1, bp_1))
                bp_2_mag_median = np.nanmedian(np.apply_along_axis(np.linalg.norm, 1, bp_2))

                if bp_1_mag_median >= bp_2_mag_median:
                    centre_point = bp_2 + ((bp_1 - bp_2) / 2)

                else:
                    centre_point = bp_1 + ((bp_2 - bp_1) / 2)
                raw_data.extend((centre_point, likelihood))

            self.body_parts.extend(pair_names)

            self.df = self.df.join(
                pd.DataFrame(
                    np.hstack(raw_data),
                    columns=pd.MultiIndex.from_product([pair_names, ['x', 'y', 'likelihood']]),
                    index=self.df.index
                )
            )

    def __repr__(self):
        return (
            f'{__class__.__name__} data_label={self.data_label}\n' \
            f'norm={self.future_scaling}, vid={self.video_path},\n' \
            f'x_max={self.x_max}, y_max={self.y_max}'
        )

    @classmethod
    def from_csv(cls, csv_path, **kwargs):
        """
        Python classmethod to import DataFrame from csv files with DeepLabCut (DLC) format

        Parameters
        ----------
        csv_filename: str
            Name of csv csv_path to be analysed; with or without csv_path-extension

        """
        if not csv_path.endswith('.csv'):
            msg = 'The argument has to be a string with the name of a csv csv_path'
            raise AttributeError(msg)
        
        path = Path(csv_path)
        if not path.exists():
            msg = 'The defined path to the .csv file does not exist'
            raise AttributeError(msg)

        # Import the csv csv_path
        type_dict = {
            'coords': int, 'x': float,
            'y': float, 'likelihood': float
        }

        df = pd.read_csv(
            csv_path, engine='c',
            index_col=0, skiprows=1, header=[0, 1],
            dtype=type_dict, na_filter=False
        )

        return cls(df, **kwargs)

    @classmethod
    def init_many(cls, file_paths, init_from='csv', labels=None, init_kwargs={}):
        """
        Method to create many dlcDF objects simultaniously using specified initialization classmethod
        """
        if init_from == 'csv':
            init_method = cls.from_csv
        else:
            msg = 'This file type has no init function implementation, currently'
            raise AttributeError(msg)

        if labels is None:
            return [init_method(file_path, **init_kwargs) for file_path in file_paths]
        else:
            return [init_method(file_path, data_label=label, **init_kwargs) for file_path, label in zip(file_paths, labels)]

    @staticmethod
    def map_function(func, dlcDF_objs, keep_label=True, manual_labels=None, kwargs_for_meth={}):
        """
        Method for applying function to a list of dlcDF objects
        """
        if manual_labels is None:
            if keep_label is True:
                return {dlcDF_obj.data_label: func(dlcDF_obj, **kwargs_for_meth) for dlcDF_obj in dlcDF_objs}
            else:
                return [func(dlcDF_obj, **kwargs_for_meth) for dlcDF_obj in dlcDF_objs]
        else:
            return {label: func(dlcDF_obj, **kwargs_for_meth) for label, dlcDF_obj in zip(manual_labels, dlcDF_objs)}
        

    def remove_flicks_hv(
        self, max_vel=100, range_thresh=100, flicks_hivel=False,
        save=False
    ):
        """
        Clean low likelihood and high velocity points from raw dataframe

        Parameters
        ----------
        max_vel: int, default 150
            The maximum velocity between two points.
            Will become automatically generated with reference to
            fps of respective video, x_max and y_max.
        range_thresh: int, default 50
        save: bool, default False
            Bool for saving/exporting the resulting dataframe to a .csv csv_path

        Returns
        -------
        new_df: pandas.DataFrame
            The cleaned df
        """
        if not isinstance(save, bool):
            msg = 'The save variable has to be bool'
            raise AttributeError(msg)

        # Remove flicks from a copy of df
        new_df = irpd(self.df.copy())

        for body_part in self.body_parts:
            """Clean flicks"""
            new_df.loc[
                get_flick_loc(new_df, body_part, max_vel),
                [(body_part, 'x'), (body_part, 'y')]] = np.nan

            """Clean high velocity values"""
            new_df.loc[
                get_hv_loc(new_df, body_part, max_vel, range_thresh),
                [(body_part, 'x'), (body_part, 'y')]] = np.nan

        if self.future_scaling:
            new_df.loc[:, (slice(None), 'x')] = \
                new_df.loc[:, (slice(None), 'x')] / self.x_max
            new_df.loc[:, (slice(None), 'y')] = \
                new_df.loc[:, (slice(None), 'y')] / self.y_max

        if save is True:
            csv_name = 'cleaned_{}'.format(self.csv_filename)
            new_df.to_csv(csv_name, sep='\t')

        return new_df

    def interpolate(self, method='linear', order=None, save=False):
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
            msg = 'The save variable has to be bool'
            raise AttributeError(msg)

        new_df = self.remove_flicks_hv()

        for body_part in self.body_parts:
            for comp in ('x', 'y'):
                new_df.loc[:, (body_part, comp)] = \
                    new_df.loc[:, (body_part, comp)].interpolate(
                        method=method, order=order,
                        limit_area='inside')

        if save is True:
            csv_name = 'interpolated_{}'.format(self.csv_filename)
            new_df.to_csv(csv_name, sep='\t')

        return new_df

    def coords(self, body_part=None, state='raw'):
        """Returns body part coordinate from"""        
        def coord_list(body_part_df):
            coords = []
            for i in range(len(body_part_df.x)):
                coords.append(
                    (body_part_df.x[i], body_part_df.y[i])
                )
            return np.array(coords)

        use_df = self.get_state(state)
        if body_part == None:
            coords = {}
            for body_part in self.body_parts:
                coords[body_part] = coord_list(use_df[body_part])
        elif isinstance(body_part, str):
            coords = coord_list(body_part)
        else:
            msg = 'body_part has to be string or None'
            raise AttributeError(msg)

        return coords

    def view(self, state='raw'):
        """For viewing data frame in terminal; don't use in jupyter notebook"""
        use_df = self.get_state(state)
        for body_part in self.body_parts:
            print(use_df[body_part])

    def get_state(self, state, **kwargs):
        if state == 'raw':
            return self.df
        elif state == 'cleaned':
            return self.remove_flicks_hv(**kwargs)
        elif state == 'interpolated':
            return self.interpolate(**kwargs)
        else:
            msg = 'The state can only be raw; cleaned; interpolated'
            raise AttributeError(msg)
