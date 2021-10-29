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

PARAMETERS_TO_COMPARE = (
    "Displacement",
    "Median_speed",
    "Median_acceleration",
    "Discrimination_index",  # Novel
    "Novelty_preference",  # Novel
    "Object_bias_score",  # Novel
)

with pd.ExcelWriter(
    RESULT_DIR / "manova.xlsx",
    engine_kwargs={
        "strings_to_formulas": False,
        "strings_to_urls": False,
    },
) as writer:
    for dataset in iglob(str(RESULT_DIR / "for_analysis" / f"*.{FILE_EXTENSION}")):
        dataset = Path(dataset)

        df = pd.read_parquet(dataset)
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

        grouped = concatenated.groupby(by="Stage")
        dfs = [value[1] for value in grouped]
        habituation, training, novelty = dfs

        experiments = pd.concat([training, novelty], axis=1, sort=True)

        for parameter in PARAMETERS_TO_COMPARE:
            print(parameter)
            analyse = MANOVA.from_formula(
                f"C(Sex) + C(HCAR1) + C(VXFAD) + C(Treatment) + C(Group) ~ {parameter}",
                concatenated,
            )
            analyse.mv_test().summary_frame.to_excel(
                writer, sheet_name=f"{parameter}_{dataset.stem}"
            )

            # tukey = pairwise_tukeyhsd(endog=voter_age,  # Data
            #                           groups=voter_race,  # Groups
            #                           alpha=0.05)  # Significance level
            #
            # tukey.plot_simultaneous()  # Plot group confidence intervals
            # plt.vlines(x=49.57, ymin=-0.5, ymax=4.5, color="red")
            #
            # tukey.summary()  # See test summary
