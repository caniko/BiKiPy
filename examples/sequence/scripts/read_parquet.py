import numpy as np
import pandas as pd

from bikipy.reader import DeepLabCutReader

index = 378  # bad
# index = 382

df = DeepLabCutReader(
    df_path=f"/mnt/BigData/oul_dataset/{index}/0.coordinates-{index}-timestamped.parquet",
    midpoint_groups={"eye_center": ("left_ear", "right_ear")},
)
print(df.augmented["eye_center"])

1
