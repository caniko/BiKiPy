from collections import defaultdict

import numpy as np
from pydantic import FilePath, validate_arguments

from bikipy.perimeter.base import (
    PerimeterSet,
    perimeter_set_from_image_name_to_perimeters,
)
from bikipy.perimeter.circle.model import (
    CircleFixedRadiusPerimeter,
    CircleVariableRadiusPerimeter,
)
from bikipy.utils.makesense import (
    get_line_endpoints_from_makesense_row,
    read_makesense_line,
    read_makesense_point,
    recording_resolution_from_makesense_row,
)


def circle_from_makesense_point(data_path: FilePath, **perimeter_kwargs) -> dict[str, PerimeterSet]:
    result = defaultdict(dict)
    for _, row in read_makesense_point(data_path).iterrows():
        result[row["image_name"]][row["label"]] = CircleVariableRadiusPerimeter(
            center_pixels=np.array([row["x"], row["y"]], dtype=np.int16),
            label=row["label"],
            recording_resolution=recording_resolution_from_makesense_row(row),
            makesense_image_name=row["image_name"],
            **perimeter_kwargs,
        )

    return perimeter_set_from_image_name_to_perimeters(result)


@validate_arguments
def circle_from_makesense_line(data_path: FilePath, **perimeter_kwargs) -> dict[str, PerimeterSet]:
    result = defaultdict(dict)
    for _, row in read_makesense_line(data_path).iterrows():
        center, edge = get_line_endpoints_from_makesense_row(row)
        result[row["image_name"]][row["label"]] = CircleFixedRadiusPerimeter(
            center_pixels=center,
            radius_pixels=np.linalg.norm((center - edge)),  # AB vector is in pixels, must be meters
            label=row["label"],
            recording_resolution=recording_resolution_from_makesense_row(row),
            makesense_image_name=row["image_name"],
            **perimeter_kwargs,
        )

    return perimeter_set_from_image_name_to_perimeters(result)
