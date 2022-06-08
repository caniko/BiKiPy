import pandas as pd
from pydantic import DirectoryPath


def initialize_metadata_data_frame(root_directory: DirectoryPath, settings: dict):
    return pd.read_excel(
        root_directory / settings["immutable"]["metadata_filename"],
        index_col="Animal",
        header=(0, 1) if settings["stageful_metadata"] else 0,
        names=["Feature", "Stage"] if settings["stageful_metadata"] else None,
    )
