from .Vector2D import Vector2D
import pandas as pd
import csv
from .analysis_settings import *
import os


class DeepLabCutCsvReader:
    def __init__(self, csv_file):
        if isinstance(csv_file, str) and csv_file.endswith(".csv"):
            self.csv_file = csv_file
        else:
            raise TypeError("The argument has to be a string linking to a csv file")

    def __call__(self, *args, **kwargs):
        data = pd.read_csv(os.path.join(CSV_DIR, self.csv_file), skiprows=2, names=self.headers)
        return data

    def analyze(self):
        with open(self.csv_file, "rb") as file:

        return

    @property
    def headers(self):
        headers = []
        for part_name in self.part_names:
            headers.append("")
            headers.append(part_name)
            headers.append(" ")
        return headers

    @property
    def part_names(self):
        column_names = pd.read_csv(os.path.join(CSV_DIR, self.csv_file), skiprows=1, nrows=1)
        part_names = [header for header in column_names][1::3]
        return part_names

    @property
    def number_of_parts(self):
        number_of_parts = len(self.part_names)
        if not number_of_parts.is_integer():
            raise AttributeError("{} is not a valid datafile".format(str(self.csv_file)))

        return number_of_parts
