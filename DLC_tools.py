import os
from Vector2D import Vector2D
import pandas as pd

from DLC_analysis_settings import *


class DLCsv:
    def __init__(self, csv_file):
        if not isinstance(csv_file, str) and not csv_file.endswith(".csv"):
            raise TypeError("The argument has to be a string with the name of a csv file")
        self.csv_file = csv_file

    def csv_reader(self, usecols=None):
        return pd.read_csv(self.csv_file, index_col=0, skiprows=1, usecols=usecols)

    def __call__(self, invert_y=True, normalize=False, *args, **kwargs):
        def usecols_gen(total=self.shape[1]):
            for i in range(1, total+1, 3):
                yield [0]+[x for x in range(i, i+3)]

        result = {}
        for body_part, columns in zip(self.body_parts, usecols_gen()):
            result[body_part] = self.csv_reader(usecols=columns)
        return result

    def normalize_data(self, body_part_centre, body_part_1, body_part_2):
        with open("angle_data.csv", 'w') as outfile:
            rows = self.shape[0]
            for row_index in range(1, rows):
                vector_centre = Vector2D(*self.bp_coords(body_part_centre, row_index))
                vector_body_part_1 = Vector2D(*self.bp_coords(body_part_1, row_index))-vector_centre
                vector_body_part_2 = Vector2D(*self.bp_coords(body_part_2, row_index))-vector_centre
                return

    def bp_coords(self, body_part, row_index, invert_y=True, normalize=False):
        row = self.__call__()[body_part].loc[str(row_index)].tolist()
        return row[0], row[1]

    @property
    def body_parts(self):
        return self.csv_reader().columns.values.tolist()[::3]

    @property
    def shape(self):
        rows, columns = self.csv_reader().shape
        assert (columns/3).is_integer(), "Data file is invalid"
        return rows, columns

    def __repr__(self):
        return "{} with {}".format(__class__.__name__, self.csv_file)
