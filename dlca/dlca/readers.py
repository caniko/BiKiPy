from dlca.video_analysis import get_video_data
from collections import deque
import pandas as pd
import numpy as np

class DLCsv:
    def __init__(self, csv_filename, normalize=False, video_file=None,
                 x_max=None, y_max=None):
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

        if normalize is True:
            msg = 'x max and y max has to defined in order to normalize'
            raise AttributeError(msg)

        self.normalize = normalize

        # Import the csv file
        self.csv_filename = csv_filename
        type_dict = {'coords': int, 'x': float,
                     'y': float, 'likelihood': float}
        self.raw_df = pd.read_csv(csv_filename, engine='c', delimiter=',',
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
            frame, self.x_max, self.y_max = get_video_data(
                filename=video_file)
        else:
            self.x_max = x_max
            self.y_max = y_max

    def __repr__(self):
        header = '{}(\"{}\"):\n'.format(__class__.__name__, self.csv_filename)

        line_i = 'norm={}, vid={},\n'.format(
            self.normalize, self.video_file)

        line_ii = 'x_max={}, y_max={}'.format(self.x_max, self.y_max)

        base = header + line_i + line_ii
        return base

    def clean(self, min_like=0.90, max_dif=50, save=False):
        if not isinstance(save, bool):
            msg = 'The save variable has to be bool'
            raise AttributeError(msg)

        def bad_coords(comp):
            original = new_df.loc[:, (body_part, comp)].values
            minus_first = np.delete(original, 0, 0)
            minus_last = np.delete(original, -1, 0)

            # Disregard warnings as they arise from NaN being compared to numbers
            np.warnings.filterwarnings('ignore')

            ele_dif = np.subtract(minus_first, minus_last)
            bad_values = deque(np.greater_equal(ele_dif, max_dif))
            bad_values.appendleft(False)
            return bad_values

        new_df = self.raw_df.copy()

        for body_part in self.body_parts:
            """Clean low likelihood values"""

            new_df.loc[new_df[(body_part, 'likelihood')] < min_like,
                       [(body_part, 'x'), (body_part, 'y')]] = np.nan

            """Clean high velocity values"""

            invalid_coords = np.logical_or(bad_coords('x'), bad_coords('y'))

            new_df.loc[invalid_coords,
                       [(body_part, 'x'), (body_part, 'y')]] = np.nan

        if self.normalize:
            new_df.loc[:, (slice(None), 'x')] = \
                new_df.loc[:, (slice(None), 'x')] / self.x_max
            new_df.loc[:, (slice(None), 'y')] = \
                new_df.loc[:, (slice(None), 'y')] / self.y_max

        if save is True:
            csv_name = 'cleaned_{}.csv'.format(self.csv_filename)
            new_df.to_csv(csv_name, sep='\t')

        return new_df

    def interpolate(self, method='slinear', order=None, save=False):
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
            csv_name = 'interpolated_{}.csv'.format(self.csv_filename)
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

    def get_state(self, state):
        state_dict = {'raw': self.raw_df,
                      'cleaned': self.clean(),
                      'interpolated': self.interpolate()}
        return state_dict[state]
