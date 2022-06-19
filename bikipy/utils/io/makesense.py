from logging import getLogger

import pandas as pd
from pydantic import FilePath

logger = getLogger(__name__)


def read_makesense_rectangle(data_path: FilePath, invert_y: bool = True) -> pd.DataFrame:
    result = pd.read_csv(
        data_path, header=None, names=("label", "x", "y", "vec_x", "vec_y", "image_name", "x_res", "y_res")
    )
    if invert_y:
        result.loc[:, "y"] = result["y_res"] - result.loc[:, "y"]
        result.loc[:, "vec_y"] = -result["vec_y"]
    return result


def read_makesense_line(data_path: FilePath, invert_y: bool = True) -> pd.DataFrame:
    result = pd.read_csv(
        data_path,
        names=("label", "1x", "1y", "2x", "2y", "image_name", "x_res", "y_res"),
    )
    if invert_y:
        result.loc[:, "1y"] = result["y_res"] - result.loc[:, "1y"]
        result.loc[:, "2y"] = result["y_res"] - result.loc[:, "2y"]
    return result


def get_line_endpoints_from_makesense_row(row: pd.Series) -> tuple:
    return row.values[1:3], row.values[3:5]


def read_first_makesense_line(data_path: FilePath) -> tuple:
    return get_line_endpoints_from_makesense_row(read_makesense_line(data_path).iloc[0])


def read_makesense_point(data_path: FilePath, invert_y: bool = True) -> pd.DataFrame:
    result = pd.read_csv(
        data_path,
        names=("label", "x", "y", "image_name", "x_res", "y_res"),
    )
    if invert_y:
        result.loc[:, "y"] = result["y_res"] - result.loc[:, "y"]
    return result


def get_point_from_makesense_row(row: pd.Series):
    return row.values[1:3]


def image_name_to_point_from_makesense(data_path: FilePath, return_first_value: bool = False):
    result = {}
    for _, row in read_makesense_point(data_path).iterrows():
        assert row["image_name"] not in result, "Reference point dataset has several reference points for one image"

        reference_point = get_point_from_makesense_row(row)

        if return_first_value:
            return reference_point

        result[row["image_name"]] = reference_point

    return result
