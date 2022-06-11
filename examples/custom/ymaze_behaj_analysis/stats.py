from pathlib import Path

import pandas as pd

from bikipy.behaviour.summary import StatisticalAnalysis

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_ANALYSIS_DIR = RESULTS_DIR / "for_analysis"
RESULTS_STATS_DIR = RESULTS_DIR / "statistics"

analysis_data = {
    0: RESULTS_ANALYSIS_DIR / "Y-maze_07.06.2020 (1A) & Y-maze (after)_26.08.2020 (2A).parquet",
    1: RESULTS_ANALYSIS_DIR / "Y-maze2_31.08.2020 (1B) & Y-maze2 (after)_25.11.2020 (2B).parquet",
}


for i in range(2):
    after_df, before_df = list(pd.read_parquet(analysis_data[i]).groupby(axis=1, level=0))
    before_df = before_df[1].droplevel(axis=1, level=0)
    after_df = after_df[1].droplevel(axis=1, level=0)
    analysis_df = before_df - after_df

    metadata_df = pd.read_excel("y-maze_metadata.xlsx", sheet_name=i).set_index("Animal ID")

    summary = StatisticalAnalysis(
        analysis_df=analysis_df,
        metadata_df=metadata_df,
        category_columns=metadata_df.columns[1:6].to_list(),
        feature_columns=("Alternations", "Spontaneous alternations"),
        root_directory_path=RESULTS_STATS_DIR,
        identifier=analysis_data[i].stem,
    )
    # summary.category_pair_combinations
    summary.categorical_to_feature_pairwise_tukey()
