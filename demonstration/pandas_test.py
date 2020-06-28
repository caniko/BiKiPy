import pandas as pd
import os

import dlca

csv_path = os.path.realpath(
    "C:/Users/Can/Projects/CINPLA/DeepLabCut/DeepLabCutAnalysis/example_data/test_tracking.csv"
)
video_path = os.path.realpath(
    "C:/Users/Can/Projects/CINPLA/DeepLabCut/DeepLabCutAnalysis/data/test_video.mp4"
)
df = dlca.KinematicData(
    csv_path,
    future_scaling=True,
    video_path=video_path,
    centre_bp=[("left_ear", "right_ear")],
)

print(df.body_parts)
