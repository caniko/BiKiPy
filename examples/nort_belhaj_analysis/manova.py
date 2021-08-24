from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA

WORKING_DIR = Path(".").resolve()
RESULT_DIR = WORKING_DIR / "results"

DATA_DIR = WORKING_DIR / "data"

FILE_EXTENSION = ".parquet"

for dataset in ("exp_2020-08-30", "exp_2020-11-23"):
    df = pd.read_parquet((RESULT_DIR / dataset).with_suffix(FILE_EXTENSION))
    df = df[
        [
            ("All", "Displacement"),
            ("All", "Median speed"),
            ("All", "Median acceleration"),
            ("Discrimination index", "Total"),
            ("Novelty preference", "Total"),
            ("Object bias score", "Total"),
        ]
    ].droplevel(0, axis=1)

    metadata_df = pd.read_excel(DATA_DIR / "nort_round_2.xlsx", index_col="Test")
    concatenated = pd.concat([df, metadata_df], axis=1, sort=True)
    concatenated.rename(columns={"HCAR1": "HCARI"})
    analyse = MANOVA.from_formula(
        "Displacement ~ Sex + HCAR1 + VXFAD + Treatment + Group",
        concatenated
    )
    print(analyse)
