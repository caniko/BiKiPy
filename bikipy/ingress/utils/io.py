from functools import lru_cache

import pandas as pd
from pydantic import DirectoryPath


@lru_cache
def initialize_metadata_data_frame(root_directory: DirectoryPath, stageful_metadata: bool):
    return pd.read_excel(
        next(root_directory.glob("metadata.*")),
        index_col=0,
        header=(0, 1) if stageful_metadata else 0,
    )
