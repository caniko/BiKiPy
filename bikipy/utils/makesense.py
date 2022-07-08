from functools import lru_cache
from logging import getLogger
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import FilePath

from bikipy import INVERT_Y_AXIS
from bikipy.core.typing import NDArrayFp64, NDArrayInt16

logger = getLogger(__name__)


def read_makesense_rectangle(data_path: FilePath, invert_y_axis: bool = INVERT_Y_AXIS) -> pd.DataFrame:
    result = pd.read_csv(
        data_path, header=None, index_col=0, names=("x", "y", "vec_x", "vec_y", "image_name", "x_res", "y_res")
    )
    if invert_y_axis:
        result.loc[:, "y"] = result["y_res"] - result.loc[:, "y"]
        result.loc[:, "vec_y"] = -result["vec_y"]
    return result


def read_makesense_line(data_path: FilePath, invert_y_axis: bool = INVERT_Y_AXIS) -> pd.DataFrame:
    result = pd.read_csv(
        data_path, names=("label", "1x", "1y", "2x", "2y", "image_name", "x_res", "y_res"), index_col=None, header=None
    )
    if invert_y_axis:
        result.loc[:, "1y"] = result["y_res"] - result.loc[:, "1y"]
        result.loc[:, "2y"] = result["y_res"] - result.loc[:, "2y"]
    return result


def get_line_endpoints_from_makesense_row(row: pd.Series) -> tuple:
    return row.values[1:3], row.values[3:5]


def read_first_makesense_line(data_path: FilePath) -> tuple:
    return get_line_endpoints_from_makesense_row(read_makesense_line(data_path).iloc[0])


@lru_cache
def read_makesense_point(data_path: FilePath, invert_y_axis: bool = INVERT_Y_AXIS) -> pd.DataFrame:
    result = pd.read_csv(
        data_path,
        names=("label", "x", "y", "image_name", "x_res", "y_res"),
    )
    if invert_y_axis:
        result.loc[:, "y"] = result["y_res"] - result.loc[:, "y"]
    return result


def get_point_from_makesense_row(row: pd.Series) -> NDArrayFp64:
    return row.values[1:3]


# Use get_only_point_from_makesense instead!
# def read_first_makesense_point(*args, **kwargs) -> NDArrayFp64:
#     return get_point_from_makesense_row(read_makesense_point(*args, **kwargs).iloc[0])


def image_name_to_point_from_makesense(data_path: FilePath, only_point: bool = True, only_stem: bool = True):
    result = {}
    for _, row in read_makesense_point(data_path).iterrows():
        key = row["image_name"]
        if only_stem:
            key = Path(key).stem
        result[key] = get_point_from_makesense_row(row) if only_point else row

    return result


def get_only_point_from_makesense(data_path: FilePath) -> NDArrayFp64:
    df = read_makesense_point(data_path)

    if np.any(df["image_name"].duplicated(keep=False)):
        msg = "Makesense point dataset has several reference points for one image, one per image was expected"
        raise ValueError(msg)

    return get_point_from_makesense_row(df.iloc[0])


def recording_resolution_from_makesense_row(row: pd.Series) -> NDArrayInt16:
    return np.array((row["x_res"], row["y_res"]), dtype=np.int16)


def image_names_from_makesense(
    data_path: FilePath, makesense_shape: Literal["rectangle", "line", "point"]
) -> pd.DataFrame:
    match makesense_shape:
        case "rectangle":
            data = read_makesense_rectangle(data_path)
        case "line":
            data = read_makesense_line(data_path)
        case "point":
            data = read_makesense_point(data_path)

    if len(data) > 1:
        logger.warning(f"first_image_name_from_makesense detected more than one perimeter in one file: {data_path}")

    return data.loc[:, "image_name"]


def first_image_name_from_makesense(*args, **kwargs) -> str:
    return image_names_from_makesense(*args, **kwargs).iloc[0]


SHAPE_TO_MAKESENSE_TYPE = {
    "circle": "line",
    "triangle": "polygon",
    "rectangle": "rectangle",
    "polygon": "polygon",
}
