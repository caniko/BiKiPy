import pandas as pd

from bikipy.behaviour.summary import StatisticalAnalysis


metadata_frame = {
    0: pd.read_excel("y-maze_metadata.xlsx", sheet_name=0).set_index("Animal ID"),
    2: pd.read_excel("y-maze_metadata.xlsx", sheet_name=1).set_index("Animal ID"),
}
metadata_frame[1] = metadata_frame[0]
metadata_frame[3] = metadata_frame[2]

summary = StatisticalAnalysis.merge_analysis_data_with_metadata(
            df, metadata_frame[i], category_columns=metadata_frame[i].columns[1:7]
        )
