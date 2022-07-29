import numpy as np
import pandas as pd

from bikipy.reader.utils import merge_timestamps_with_dlc


def file_to_timeseries_sequence(file_path):
    datetime_array = (
        pd.read_csv(file_path, header=None, usecols=[16], parse_dates=[0]).values.T[0].astype(np.datetime64)
    )
    return datetime_array - datetime_array[0]


merge_timestamps_with_dlc("/home/can/Projects/bikipy/examples/sequence/OUL/dataset", file_to_timeseries_sequence)
