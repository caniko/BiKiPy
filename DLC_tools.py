from Vector2D import Vector2D
import pandas as pd
import re
import matplotlib.pylab as plt

from DLC_analysis_settings import *


class DLCsv:
    def __init__(self, csv_file, invert_y=True, normalize=False):
        if not isinstance(csv_file, str) and not csv_file.endswith(".csv"):
            msg = "The argument has to be a string with the name of a csv file"
            raise TypeError(msg)
        self.csv_file = csv_file
        self.invert_y = invert_y
        self.x_max, self.y_max = [500, 300]
        if normalize:
            self.normalize = True
        else:
            self.normalize = False

    @property
    def two_cm_upper_limit(self):
        if self.invert_y:
            if self.normalize:
                return 1 / 1.18
            else:
                return self.y_max / 1.18
        else:
            if self.normalize:
                return 1 / 4.65
            else:
                return self.y_max / 4.65

    @property
    def two_cm_lower_limit(self):
        if self.invert_y:
            if self.normalize:
                return 1 / 4.65
            else:
                return self.y_max / 4.65
        else:
            if self.normalize:
                return 1 / 1.18
            else:
                return self.y_max / 1.18

    def position_preference(self, plot=False):
        total_frames = self.shape[0]-1

        nose_y = self.df['nose']['y'].values.tolist()[1:]
        left_ear_y = self.df['left_ear']['y.2'].values.tolist()[1:]
        right_ear_y = self.df['right_ear']['y.3'].values.tolist()[1:]
        lower_environment = 0
        upper_environment = 0
        for i in range(total_frames):
            if left_ear_y[i] > self.two_cm_lower_limit \
                    and nose_y[i] > self.two_cm_lower_limit:
                lower_environment += 1
            elif right_ear_y[i] > self.two_cm_lower_limit \
                    and nose_y[i] > self.two_cm_lower_limit:
                lower_environment += 1
            elif left_ear_y[i] < self.two_cm_upper_limit \
                    and nose_y[i] < self.two_cm_upper_limit:
                upper_environment += 1
            elif right_ear_y[i] < self.two_cm_upper_limit \
                    and nose_y[i] < self.two_cm_upper_limit:
                upper_environment += 1

        percent_upper = (upper_environment / total_frames) * 100
        percent_lower = (lower_environment / total_frames) * 100
        if plot:
            pass
        else:
            return ("Lower environment: {}% \nUpper environment: {}%".
                    format(round(percent_lower, 2), round(percent_upper, 2)))

    def __call__(self, *args, **kwargs):
        pass

    def bp_coords(self, body_part, row_index):
        row = self.df[body_part].loc[str(row_index)].tolist()
        return row[0], row[1]

    def read_csv(self, usecols=None):
        return pd.read_csv(self.csv_file, index_col=0,
                           skiprows=2, usecols=usecols)

    def angle_df(self, body_part_centre, body_part_1, body_part_2):
        """Return a csv with the angle between three body parts per frame
        WIP
        """
        with open("angle_data.csv", 'w') as outfile:
            rows = self.shape[0]
            for row_index in range(1, rows):
                vector_centre = Vector2D(*self.bp_coords(
                    body_part_centre, row_index))
                vector_body_part_1 = Vector2D(*self.bp_coords(
                    body_part_1, row_index))-vector_centre
                vector_body_part_2 = Vector2D(*self.bp_coords(
                    body_part_2, row_index))-vector_centre
                return

    @property
    def df(self):
        def usecols_gen(total=self.shape[1]):
            for i in range(1, total+1, 3):
                yield [0]+[x for x in range(i, i+3)]

        if self.normalize:
            def y_normalizer(y_coord):
                if self.invert_y:
                    return 1 - y_coord / self.y_max
                else:
                    return y_coord / self.y_max

        result = {}
        for body_part, columns in zip(self.body_parts, usecols_gen()):
            body_part_df = self.read_csv(usecols=columns)
            if self.normalize:
                for column in body_part_df:
                    if re.match(r'x(\.\d)?', column):  # or re.search() for
                        # partial matches
                        body_part_df[column] = body_part_df[column].apply(
                            lambda x_coord: x_coord / self.x_max)
                    if re.match(r'y(\.\d)?', column):
                        body_part_df[column] = body_part_df[column].apply(
                            y_normalizer)     # Inspector warning irrelevant
            elif self.invert_y:
                for column in body_part_df:
                    if re.match(r'x(\.\d)?', column):
                        body_part_df[column] = body_part_df[column].apply(
                            lambda y_coord: self.y_max - y_coord)
            result[body_part] = body_part_df
        return result

    @property
    def body_parts(self):
        body_part_row = pd.read_csv(self.csv_file, index_col=0,
                                    skiprows=1, nrows=1)
        return body_part_row.columns.values.tolist()[::3]

    @property
    def shape(self):
        rows, columns = self.read_csv().shape
        assert (columns/3).is_integer(), "Data file is invalid"
        return rows, columns

    def __repr__(self):
        return "{} with {}".format(__class__.__name__, self.csv_file)
