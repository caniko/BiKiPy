import numpy as np
import pandas as pd

df = pd.read_parquet("/mnt/BigData/oul_dataset/136/2.coordinates-136.parquet")

datetime_array = (
    pd.read_csv("/mnt/BigData/oul_dataset/136/2.timestamps-136.csv", header=None, usecols=[16], parse_dates=[0])
    .values.T[0]
    .astype(np.datetime64)
)
timestamp = (datetime_array - datetime_array[0]).astype(float) / 10**6

df.index = datetime_array
1
