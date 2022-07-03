import numpy as np
import pandas as pd

from bikipy.reader.utils import merge_timestamps_with_dlc


def file_to_timeseries_sequence(file_path):
    datetime_arr = pd.read_csv(file_path, header=None, usecols=[16], parse_dates=[0]).values.T[0]
    array = np.diff(datetime_arr).astype(np.timedelta64).astype(int)

    return np.concatenate(([0], array))


merge_timestamps_with_dlc("/home/can/Projects/bikipy/examples/sequence/OUL/dataset", file_to_timeseries_sequence)
