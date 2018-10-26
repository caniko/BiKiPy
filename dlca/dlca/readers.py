import pandas as pd
import numpy as np
import os

from dlca.video_analysis import get_video_data
from dlca.mechanics import high_velocity_values


class DLCsv:
    def __init__(self, csv_filename, normalize=False, video_file=None,
                 x_max=None, y_max=None, path=os.getcwd()):
        """
        Python class to analyze csv files from DeepLabCut (DLC)
        Parameters
        ----------
        csv_filename: str
            Name of csv file to be analysed; with or without file-extension
        normalize: boolean, default False
            Normalizes the coordinates. Requires x_max and y_max to be defined
        x_max: {None, int}, default None
            Maximum x value, can be extracted from video sample or defined
        y_max: {None, int}, default None
            Maximum y value, can be extracted from video sample or defined
        """
        if not isinstance(csv_filename, str) and not csv_filename.endswith(
                '.csv'):
            msg = 'The argument has to be a string with the name of a csv file'
            raise AttributeError(msg)

        if not (isinstance(x_max, (int, float, type(None))) and
                isinstance(y_max, (int, float, type(None)))):
            msg = 'x and y max are integers; not {}; {}'.format(x_max, y_max)
            raise AttributeError(msg)

        if normalize is True and x_max is None and y_max is None:
            msg = 'x max and y max has to defined in order to normalize'
            raise AttributeError(msg)

        self.normalize = normalize
        self.path = os.path.abspath(path)

        # Import the csv file
        self.csv_filename = csv_filename
        self.csv_file_path = os.path.join(self.path, csv_filename)
        type_dict = {'coords': int, 'x': float,
                     'y': float, 'likelihood': float}
        self.raw_df = pd.read_csv(self.csv_file_path, engine='c',
                                  index_col=0, skiprows=1, header=[0, 1],
                                  dtype=type_dict, na_filter=False)

        # Get name of body parts
        csv_multi_i = list(self.raw_df)
        body_parts = []
        for i in range(0, len(csv_multi_i), 3):
            body_parts.append(csv_multi_i[i][0])
        self.body_parts = body_parts

        self.video_file = video_file
        self.vid_test = video_file is True or isinstance(video_file, str)
        if self.vid_test:
            self.x_max, self.y_max = get_video_data(
                filename=video_file)[1:3]
        else:
            self.x_max = x_max
            self.y_max = y_max

    def __repr__(self):
        header = '{}(\"{}\"):\n'.format(
            __class__.__name__, self.csv_filename if self.path == os.getcwd()
            else self.csv_file_path)

        line_i = 'norm={}, vid={},\n'.format(
            self.normalize, self.video_file)

        line_ii = 'x_max={}, y_max={}'.format(self.x_max, self.y_max)

        base = header + line_i + line_ii
        return base

    def clean(self, min_like=0.90, max_vel=100, range_thresh=100,
              save=False):
        """Clean low likelihood and high velocity points from raw dataframe
        Parameters
        ----------
        min_like: float, default 0.90
            The minimum likelihood the coordinates of the respective row.
            If below the values, the coords are discarded while being replaced
            by numpy.NaN
        max_vel: int, default 150
            The maximum velocity between two points.
            Will become automatically generated with reference to
            fps of respective video, x_max and y_max.
        range_thresh: int, default 50
        save: bool, default False
            Bool for saving/exporting the resulting dataframe to a .csv file
        Returns
        -------
        new_df: pandas.DataFrame
            The cleaned raw_df
        """
        if not isinstance(save, bool):
            msg = 'The save variable has to be bool'
            raise AttributeError(msg)

        new_df = self.raw_df.copy()

        for body_part in self.body_parts:
            """Clean low likelihood values"""

            new_df.loc[new_df[(body_part, 'likelihood')] < min_like,
                       [(body_part, 'x'), (body_part, 'y')]] = np.nan

            """Clean high velocity values"""
            new_df.loc[
                high_velocity_values(new_df, body_part, max_vel, range_thresh),
                [(body_part, 'x'), (body_part, 'y')]] = np.nan

        if self.normalize:
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
            Bool for saving/exporting the resulting dataframe to a .csv file
        Returns
        -------
        new_df: pandas.DataFrame
            The interpolated raw_df
        """
        if not isinstance(save, bool):
            msg = 'The save variable has to be bool'
            raise AttributeError(msg)

        new_df = self.clean()

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

    def bp_coords(self, body_part, row_index, state='interpolated'):
        """Returns body part coordinate from"""
        if not isinstance(body_part, str):
            msg = 'body_part has to be string'
            raise AttributeError(msg)
        if row_index % 1 != 0:
            msg = 'row_index must be an integer'
            raise AttributeError(msg)

        use_df = self.get_state(state)
        row = use_df[body_part].loc[row_index].tolist()
        return row[0], row[1]

    def view(self, state='raw'):
        """For viewing data frame in terminal; don't use in jupyter notebook"""
        use_df = self.get_state(state)
        for body_part in self.body_parts:
            print(use_df[body_part])

    def get_state(self, state, **kwargs):
        state_dict = {'raw': self.raw_df,
                      'cleaned': self.clean(**kwargs),
                      'interpolated': self.interpolate(**kwargs)}
        return state_dict[state]


def csv_iterator(method, analysis_initi=None, state='cleaned',
                 path=os.getcwd(), ret_obj='dict',
                 kwargs_for_csv={}, kwargs_for_initi={}, kwargs_for_meth={}):
    path = os.path.abspath(path)
    csv_list = [file for file in os.listdir(path) if
                file.endswith('.csv')]
    result = {}
    for file in csv_list:
        file_path = os.path.join(path, file)
        csv_file_df = DLCsv(file_path).get_state(state, **kwargs_for_csv)
        cleaned_name = file[:-4]

        if analysis_initi is not None:
            analysis = analysis_initi(csv_file_df, **kwargs_for_initi)
            result[cleaned_name] = getattr(analysis, method)(**kwargs_for_meth)
        else:
            result[cleaned_name] = method(csv_file_df, **kwargs_for_meth)

    if ret_obj == 'dict':
        return result


