import pandas as pd
import numpy as np
import os

from dlca.functions.hviva import (
    get_high_velocity_locations as get_hv_loc,
    get_flicks_locations as get_flick_loc,
    invalid_relative_part_distance as irpd
)


class DLCsv:
    def __init__(
        self, csv_filename, centre_bp=None, future_scaling=False, min_like=0.95,
        video_file=None, x_max=None, y_max=None, path=os.getcwd()):
        """
        Python class to analyze csv files from DeepLabCut (DLC)
        Parameters
        ----------
        csv_filename: str
            Name of csv csv_path to be analysed; with or without csv_path-extension
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
        if not isinstance(csv_filename, str) and not csv_filename.endswith(
                '.csv'):
            msg = 'The argument has to be a string with the name of a csv csv_path'
            raise AttributeError(msg)

        if not (isinstance(x_max, (int, float, type(None))) and
                isinstance(y_max, (int, float, type(None)))):
            msg = 'x and y max are integers; not {}; {}'.format(x_max, y_max)
            raise AttributeError(msg)

        if future_scaling is True and video_file is None and (x_max is None and y_max is None):
            msg = 'x max and y max, or vieo_file has to defined in order to future_scaling'
            raise AttributeError(msg)

        if centre_bp is not None and not (isinstance(centre_bp, list) or isinstance(centre_bp, tuple)):
            msg = 'centre_bp can only be defined as list or tuple;\nthe most convinient data structure for body_part names'
            raise AttributeError(msg)


        self.name = csv_filename.split('\\')[-1].split('_')[0]
        self.future_scaling = future_scaling
        self.path = os.path.abspath(path)
        self.min_like = min_like

        # Import the csv csv_path
        self.csv_filename = csv_filename
        self.csv_file_path = os.path.join(self.path, csv_filename)
        type_dict = {
            'coords': int, 'x': float,
            'y': float, 'likelihood': float
        }

        self.raw_df = pd.read_csv(
            self.csv_file_path, engine='c',
            index_col=0, skiprows=1, header=[0, 1],
            dtype=type_dict, na_filter=False
        )

        # Get name of body parts
        csv_multi_i = list(self.raw_df)
        body_parts = []
        for i in range(0, len(csv_multi_i), 3):
            body_parts.append(csv_multi_i[i][0])
        self.body_parts = body_parts

        self.video_file = video_file
        self.vid_test = video_file is True or isinstance(video_file, str)
        if self.vid_test:
            try:
                from dlca.video_analysis import handle_video_data
            except ModuleNotFoundError:
                msg = 'opencv-python is required to analyse video'
                raise ModuleNotFoundError(msg)

            self.x_max, self.y_max = handle_video_data(
                video_path=video_file)[1:3]
        else:
            self.x_max = x_max
            self.y_max = y_max

        self.invalid_points = {}
        self.valid_points = {}
        for body_part in self.body_parts:
            self.invalid_points[body_part] = self.raw_df[(body_part, 'likelihood')] < self.min_like
            self.valid_points[body_part] = np.logical_not(self.invalid_points[body_part])
            
            # Remove likelihood values below min_like value
            self.raw_df.loc[
                self.invalid_points[body_part],
                [(body_part, 'x'), (body_part, 'y')]
            ] = np.nan

            # Invert y axis
            self.raw_df[(body_part, 'y')] = self.raw_df[(body_part, 'y')].map(lambda y: self.y_max - y)

        if centre_bp:
            pair_names = []
            raw_data = []
            for bp_pairs in centre_bp:
                if not isinstance(bp_pairs, tuple) and not isinstance(bp_pairs, list):
                    msg = 'The pairs must be stored in python tuple or list'
                    raise AttributeError(msg)

                if not (
                    (bp_pairs[0] in self.body_parts and bp_pairs[1] in self.body_parts) and 
                    (isinstance(bp_pairs[0], str) and isinstance(bp_pairs[1], str))):
                    msg = 'The body part names must be referred to with their names in string form'
                    raise AttributeError(msg)

                if len(bp_pairs) != 2:
                    msg = 'Each pair can only consist of 2 elements'
                    raise AttributeError(msg)

                name = 'c_%s_%s' % bp_pairs
                pair_names.append(name)

                likelihood = (self.raw_df.loc[:, [(bp_pairs[0], 'likelihood')]].values \
                            + self.raw_df.loc[:, [(bp_pairs[1], 'likelihood')]].values) \
                            / 2
                self.invalid_points[name] = likelihood < self.min_like
                self.valid_points[name] = np.logical_not(self.invalid_points[name])

                bp_1 = self.raw_df.loc[:, [(bp_pairs[0], 'x'), (bp_pairs[0], 'y')]].values
                bp_2 = self.raw_df.loc[:, [(bp_pairs[1], 'x'), (bp_pairs[1], 'y')]].values

                bp_1_mag_median = np.nanmedian(np.apply_along_axis(np.linalg.norm, 1, bp_1))
                bp_2_mag_median = np.nanmedian(np.apply_along_axis(np.linalg.norm, 1, bp_2))

                if bp_1_mag_median >= bp_2_mag_median:
                    centre_point = bp_2 + ((bp_1 - bp_2) / 2)

                else:
                    centre_point = bp_1 + ((bp_2 - bp_1) / 2)
                raw_data.extend((centre_point, likelihood))

            self.raw_df = self.raw_df.join(pd.DataFrame(
                np.hstack(raw_data),
                columns=pd.MultiIndex.from_product([pair_names, ['x', 'y', 'likelihood']]),
                index=self.raw_df.index)
            )
            self.body_parts.extend(pair_names)


    def __repr__(self):
        header = '{}(\"{}\"):\n'.format(
            __class__.__name__, self.csv_filename if self.path == os.getcwd()
            else self.csv_file_path)

        line_i = 'norm={}, vid={},\n'.format(
            self.future_scaling, self.video_file)

        line_ii = 'x_max={}, y_max={}'.format(self.x_max, self.y_max)

        base = header + line_i + line_ii
        return base

    def remove_flicks_hv(
        self, max_vel=100, range_thresh=100, flicks_hivel=False,
        save=False):
        """Clean low likelihood and high velocity points from raw dataframe
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
            The cleaned raw_df
        """
        if not isinstance(save, bool):
            msg = 'The save variable has to be bool'
            raise AttributeError(msg)

        # Remove flicks from a copy of raw_df
        new_df = irpd(self.raw_df.copy())

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
            The interpolated raw_df
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
            return self.raw_df
        elif state == 'cleaned':
            return self.remove_flicks_hv(**kwargs)
        elif state == 'interpolated':
            return self.interpolate(**kwargs)
        else:
            msg = 'The state can only be raw; cleaned; interpolated'
            raise AttributeError(msg)


def csv_iterator_pre(csv_paths, csv_file_id_depth=2, kwargs_for_csv={}):
    result = {}
    for csv_path in csv_paths:
        csv_file_df = DLCsv(csv_path, **kwargs_for_csv)

        name = csv_path.split('\\\\')[-1][:-4]
        result[name] = csv_file_df
    return result


def csv_iterator(
    method, csv_dlcsv_objs, loc_base_hierarchy=True,
    kwargs_for_meth={}):

    result = {}
    for dlcsv_obj in csv_dlcsv_objs:
        result[dlcsv_obj.name] = method(dlcsv_obj, **kwargs_for_meth)

    return result
