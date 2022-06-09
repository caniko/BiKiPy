from logging import getLogger

import pandas as pd
from pydantic import FilePath

logger = getLogger(__name__)


def read_makesense_line(data_path: FilePath) -> pd.DataFrame:
    return pd.read_csv(
        data_path,
        names=("label", "1x", "1y", "2x", "2y", "image_name", "x_res", "y_res"),
    )


def get_line_endpoints_from_makesense_row(row: pd.Series) -> tuple:
    return row.values[1:3], row.values[3:5]


def read_first_makesense_line(data_path: FilePath) -> tuple:
    return get_line_endpoints_from_makesense_row(read_makesense_line(data_path).iloc[0])


def read_makesense_point(data_path: FilePath) -> pd.DataFrame:
    return pd.read_csv(
        data_path,
        names=("label", "x", "y", "image_name", "x_res", "y_res"),
    )


def get_point_from_makesense_row(row: pd.Series):
    return row.values[1:3]


def image_name_to_point_from_makesense(data_path: FilePath, return_first_value: bool = False):
    result = {}
    for _, row in read_makesense_point(data_path).iterrows():
        assert row["image_name"] not in result, "Reference point dataset has several reference points for one image"

        reference_point = get_point_from_makesense_row(row)

        if return_first_value:
            return reference_point

        result["image_name"] = reference_point

    return result
