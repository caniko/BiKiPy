import numpy as np
import pandas as pd
from numpy import timedelta64

df = pd.read_csv("video_frame_timestamps.csv", header=None, usecols=[16], parse_dates=[0])
frame_timedelta = np.diff(df.values.T[0])[1:].astype(timedelta64).astype(int) / 1000000

frames = 0
cumulative = 0.0
frame_second_times = []
for frame_sec in frame_timedelta:
    cumulative += frame_sec
    frames += 1
    if cumulative >= 1:
        frame_second_times.append(frames)
        frames = 0
        cumulative = 0.0
print(np.mean(frame_second_times))
