from glob import glob

import numpy as np
import pandas as pd

for p in glob("dataset/**/*timestamped*"):
    df = pd.read_parquet(p)
    df.index = np.cumsum(df.index) / 1000000
    df.to_parquet(p)
