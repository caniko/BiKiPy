from glob import iglob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

WORKING_DIR = Path(".").resolve()
RESULT_DIR = WORKING_DIR / "results"

DATA_DIR = WORKING_DIR / "data"

FILE_EXTENSION = "parquet"

MOTION_PARAMETERS = ("Displacement", "Median_speed", "Median_acceleration")

PARAMETERS_TO_COMPARE = (
    ("Training", "Seconds observing", ""),
    ("Novelty", "Discrimination index", ""),
    ("Novelty", "Novelty preference", ""),
    ("Novelty", "Object bias score", ""),
)
meta_dfs = (
    pd.read_excel(DATA_DIR / "nort_round_1.xlsx", index_col="Test", sheet_name="NORT_02.06.2020"),
    pd.read_excel(
        DATA_DIR / "nort_round_1.xlsx",
        index_col="Test",
        sheet_name="NORT_ 24.08.2020 (after)",
    ),
    pd.read_excel(DATA_DIR / "nort_round_2.xlsx", index_col="Test", sheet_name="NORT2_30.08.20"),
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
    for i, dataset in enumerate(iglob(str(RESULT_DIR / "for_analysis" / f"*.{FILE_EXTENSION}"))):
        print(i)
        dataset = Path(dataset)
        df = pd.read_parquet(dataset)

        metadata_df = meta_dfs[i]
        metadata_df = metadata_df.drop_duplicates("Animal").set_index("Animal")
        concatenated = pd.concat((df, metadata_df), axis=1)

        concatenated = concatenated[~concatenated["Treatment"].isna()]
        concatenated["HCAR1"] = concatenated["HCAR1"].map(lambda x: x.strip())

        for parameter in PARAMETERS_TO_COMPARE:
            print(parameter)

            # analyse = MANOVA.from_formula(
            #     f"C(Sex) + C(HCAR1) + C(VXFAD) + C(Treatment) + C(Group) ~ {parameter}",
            #     novelty,
            # )
            # analyse.mv_test().summary_frame.to_excel(
            #     writer, sheet_name=f"{parameter}_{dataset.stem}"
            # )

            cat_plot = sb.catplot(x="Group", y=parameter, kind="violin", inner=None, data=concatenated)
            sb.swarmplot(
                x="Group",
                y=parameter,
                color="k",
                size=3,
                data=concatenated,
                ax=cat_plot.ax,
            )
            plt.savefig(RESULT_DIR / "figures" / f"{dataset.stem}_{i}_{parameter}.png")

            dfs = []
            for category in ("Sex", "HCAR1", "VXFAD", "Treatment", "Group"):
                print(category)
                tukey = pairwise_tukeyhsd(
                    endog=concatenated[parameter],  # Data
                    groups=concatenated[category],  # Groups
                    alpha=0.05,  # Significance
                )
                dfs.append(
                    pd.DataFrame(
                        data=tukey._results_table.data[1:],
                        columns=tukey._results_table.data[0],
                    )
                )
            pd.concat(dfs).to_excel(writer, sheet_name=f"{dataset.stem}_{i}_{parameter}")
