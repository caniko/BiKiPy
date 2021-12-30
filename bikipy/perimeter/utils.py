from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import FilePath


@lru_cache(50)
def reference_point_from_coco_path(
    metadata_path: Optional[FilePath], single_row: bool = True
):
    coco_data = pd.read_csv(
        metadata_path,
        names=("label", "x", "y", "image_name", "x_res", "y_res"),
    )
    if single_row:
        assert len(coco_data) == 1
        return coco_data.iloc[0].values[1:3].astype(np.float64)
    return {csv_row[3]: csv_row[1:3].astype(np.float64) for csv_row in coco_data.values}
