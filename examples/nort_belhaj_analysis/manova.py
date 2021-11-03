from glob import iglob
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

WORKING_DIR = Path(".").resolve()
RESULT_DIR = WORKING_DIR / "results"

DATA_DIR = WORKING_DIR / "data"

FILE_EXTENSION = "parquet"

MOTION_PARAMETERS = ("Displacement", "Median_speed", "Median_acceleration")

PARAMETERS_TO_COMPARE = (
    "Discrimination_index",  # Novel
    "Novelty_preference",  # Novel
    "Object_bias_score",  # Novel
)
meta_dfs = (
    pd.read_excel(
        DATA_DIR / "nort_round_1.xlsx", index_col="Test", sheet_name="NORT_02.06.2020"
    ),
    pd.read_excel(
        DATA_DIR / "nort_round_1.xlsx",
        index_col="Test",
        sheet_name="NORT_ 24.08.2020 (after)",
    ),
    pd.read_excel(
        DATA_DIR / "nort_round_2.xlsx", index_col="Test", sheet_name="NORT2_30.08.20"
    ),
    pd.read_excel(
        DATA_DIR / "nort_round_2.xlsx",
        index_col="Test",
        sheet_name="NORT2_23.11.2020 (after)",
    ),
)


with pd.ExcelWriter(
    RESULT_DIR / "manova.xlsx",
    engine_kwargs={
        "strings_to_formulas": False,
        "strings_to_urls": False,
    },
) as writer:
    for i, dataset in enumerate(
        iglob(str(RESULT_DIR / "for_analysis" / f"*.{FILE_EXTENSION}"))
    ):
        print(i)
        dataset = Path(dataset)

        df = pd.read_parquet(dataset)
        df_motion = df[
            [
                ("All", "Stage"),
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

        metadata_df = meta_dfs[i]
        del metadata_df["Stage"]
        concatenated = pd.concat([df_motion, df_nort, metadata_df], axis=1, sort=True)
        concatenated = concatenated[~concatenated["Treatment"].isna()]

        habituation = concatenated[concatenated["Stage"] == "habituation"]
        training = concatenated[concatenated["Stage"] == "training"]
        novelty = concatenated[concatenated["Stage"] == "novelty"]

        experiments = pd.concat([training, novelty], axis=1, sort=True)

        for parameter in PARAMETERS_TO_COMPARE:
            print(parameter)
            analyse = MANOVA.from_formula(
                f"C(Sex) + C(HCAR1) + C(VXFAD) + C(Treatment) + C(Group) ~ {parameter}",
                novelty,
            )
            analyse.mv_test().summary_frame.to_excel(
                writer, sheet_name=f"{parameter}_{dataset.stem}"
            )
            for category in ("Sex", "HCAR1", "VXFAD", "Treatment", "Group"):
                print(category)
                tukey = pairwise_tukeyhsd(
                    endog=novelty[parameter],  # Data
                    groups=novelty[category],  # Groups
                    alpha=0.05,  # Significance
                )
