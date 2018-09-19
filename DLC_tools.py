import os
from Vector2D import Vector2D
import pandas as pd
import numpy as np

from DLC_analysis_settings import *


class DLCsv:
    def __init__(self, csv_file, invert_y=True, normalize=False):
        if not isinstance(csv_file, str) and not csv_file.endswith(".csv"):
            raise TypeError("The argument has to be a string with the name of a csv file")
        self.csv_file = csv_file
        self.invert_y = invert_y
        self.normalize = normalize

    def __call__(self, *args, **kwargs):
        pass

    def bp_coords(self, body_part, row_index):
        row = self.df[body_part].loc[str(row_index)].tolist()
        return row[0], row[1]

    def read_csv(self, usecols=None):
        return pd.read_csv(self.csv_file, index_col=0, skiprows=2, usecols=usecols)

    def angle_df(self, body_part_centre, body_part_1, body_part_2):
        """Return a csv with the angle between three body parts per frame"""
        with open("angle_data.csv", 'w') as outfile:
            rows = self.shape[0]
            for row_index in range(1, rows):
                vector_centre = Vector2D(*self.bp_coords(body_part_centre, row_index))
                vector_body_part_1 = Vector2D(*self.bp_coords(body_part_1, row_index))-vector_centre
                vector_body_part_2 = Vector2D(*self.bp_coords(body_part_2, row_index))-vector_centre
                return

    @property
    def df(self):
        def usecols_gen(total=self.shape[1]):
            for i in range(1, total+1, 3):
                yield [0]+[x for x in range(i, i+3)]

        if self.normalize:
            x_max, y_max = self.normalize

        result = {}
        for body_part, columns in zip(self.body_parts, usecols_gen()):
            body_part_df = self.read_csv(usecols=columns)
            if self.normalize:
                body_part_df[body_part_df['b'].apply('')]
            result[body_part] = body_part_df
        return result

    @property
    def body_parts(self):
        body_part_row = pd.read_csv(self.csv_file, index_col=0, skiprows=1, nrows=1)
        return body_part_row.columns.values.tolist()[::3]

    @property
    def shape(self):
        rows, columns = self.read_csv().shape
        assert (columns/3).is_integer(), "Data file is invalid"
        return rows, columns

    def __repr__(self):
        return "{} with {}".format(__class__.__name__, self.csv_file)
