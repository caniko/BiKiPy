from pathlib import Path

import pandas as pd

WORKING_DIR = Path(".").resolve()
RESULT_DIR = WORKING_DIR / "results"

DATA_DIR = WORKING_DIR / "data"

FILE_EXTENSION = ".parquet"

for dataset in ("exp_2020-08-30", "exp_2020-11-23"):
    df = pd.read_parquet((RESULT_DIR / dataset).with_suffix(FILE_EXTENSION))
    metadata_df = pd.read_excel(DATA_DIR / "nort_round_2.xlsx", index_col="Test")
    concatenated = pd.concat([df, metadata_df], axis=1, sort=True)
    print(df)
