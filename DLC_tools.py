from Vector2D import Vector2D
import pandas as pd
import re

from DLC_analysis_settings import *


class DLCsv:
    def __init__(self, csv_file, invert_y=True, normalize=False):
        if not isinstance(csv_file, str) and not csv_file.endswith(".csv"):
            msg = "The argument has to be a string with the name of a csv file"
            raise TypeError(msg)
        self.csv_file = csv_file
        self.invert_y = invert_y
        self.normalize = normalize

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
                    return 1 - y_coord / y_max
                else:
                    return y_coord / y_max

            x_max, y_max = self.normalize

        result = {}
        for body_part, columns in zip(self.body_parts, usecols_gen()):
            body_part_df = self.read_csv(usecols=columns)
            if self.normalize:
                for column in body_part_df:
                    if re.match(r'x(\.\d)?', column):  # or re.search() for
                        # partial matches
                        body_part_df[column] = body_part_df[column].apply(
                            lambda x_coord: x_coord / x_max)
                    if re.match(r'y(\.\d)?', column):
                        body_part_df[column] = body_part_df[column].apply(
                            y_normalizer)     # Inspector warning irrelevant
            elif self.invert_y:
                for column in body_part_df:
                    if re.match(r'x(\.\d)?', column):
                        body_part_df[column] = body_part_df[column].apply(
                            lambda y_coord: y_max - y_coord)
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
