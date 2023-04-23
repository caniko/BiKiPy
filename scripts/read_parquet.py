import os
from pathlib import Path

import pandas as pd

from bikipy.utils.constants import TO_PARQUET_KWARGS

ROOT = Path("/mnt/BigData/Chrys_Behavior_Analysis/Chrys_Behavior/")

for f in ROOT.glob("**/**/*.h5"):
    df: pd.DataFrame = pd.read_hdf(f)
    # print(f.with_suffix(".parquet"))
    df.to_parquet(f.with_suffix(".parquet"), **TO_PARQUET_KWARGS)

    os.remove(f)
