from pathlib import Path

import pandas as pd
from statsmodels.multivariate.manova import MANOVA

WORKING_DIR = Path(".").resolve()
RESULT_DIR = WORKING_DIR / "results"

DATA_DIR = WORKING_DIR / "data"

FILE_EXTENSION = ".parquet"

for dataset in ("exp_2020-08-30", "exp_2020-11-23"):
    df = pd.read_parquet((RESULT_DIR / dataset).with_suffix(FILE_EXTENSION))
    df_motion = df[
        [
            ("All", "Displacement"),
            ("All", "Median_speed"),
            ("All", "Median_acceleration"),
        ]
    ].droplevel(0, axis=1)

    df_nort = df[
        [
            ("Discrimination_index", "Total"),
            ("Novelty_preference", "Total"),
            ("Object_bias_score", "Total"),
        ]
    ].droplevel(1, axis=1)

    metadata_df = pd.read_excel(DATA_DIR / "nort_round_2.xlsx", index_col="Test")
    concatenated = pd.concat([df_motion, df_nort, metadata_df], axis=1, sort=True)
    analyse = MANOVA.from_formula(
        "C(Sex) + C(HCAR1) + C(VXFAD) + C(Treatment) + C(Group) ~ Displacement",
        # "+ Median_speed "
        # "+ Median_acceleration "
        # "+ Discrimination_index "
        # "+ Novelty_preference "
        # "+ Object_bias_score "
        concatenated,
    )
    test = analyse.mv_test()
    print(analyse)
