import json
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from pydantic import DirectoryPath, FilePath
from pydantic_numpy.typing import Np2DArrayFp64

from bikipy import runtime_settings
from bikipy.utils.image import read_image_from_path
from bikipy.utils.makesense import (
    image_name_to_point_from_makesense,
    read_makesense_rectangle,
)

if TYPE_CHECKING:
    from bikipy.perimeter.base import PerimeterSet

logger = getLogger(__name__)


def init_polygon_from_makesense_coco_polygon(
    data_path: FilePath,
    image_root: Optional[DirectoryPath] = None,
    reference_point_csv_path: Optional[FilePath] = None,
    reference_point_array: Optional[Np2DArrayFp64] = None,
    invert_y_axis: bool = runtime_settings.matplotlib_invert_y_axis,
    **perimeter_kwargs,
) -> dict[str, "PerimeterSet"]:
    from bikipy.perimeter.base import perimeter_set_from_image_name_to_perimeters
    from bikipy.perimeter.polygon.base import init_polygon

    logger.debug("Generating BasePolygonPerimeter from makesense polygon data in coco format")

    with open(data_path, "rb") as in_json:
        coco = json.load(in_json)

    assert not image_root or (image_root := Path(image_root)).exists()

    # The coco annotations are not sorted with respect to the category IDs
    coco["annotations"] = sorted(coco["annotations"], key=lambda dictionary: dictionary["category_id"])

    # We don't need to do this, but better to be on the safe side
    coco["categories"] = sorted(coco["categories"], key=lambda dictionary: dictionary["id"])

    if reference_point_csv_path:
        image_name_to_reference_point = image_name_to_point_from_makesense(reference_point_csv_path)

    result = defaultdict(dict)
    for annotation in coco["annotations"]:
        current_kwargs = {}
        image_index = annotation["image_id"] - 1
        image_name = coco["images"][image_index]["file_name"]
        label = coco["categories"][annotation["category_id"] - 1]["name"]

        if reference_point_array is None:
            reference_point_array = image_name_to_reference_point[image_name] if reference_point_csv_path else None

        y_res = float(coco["images"][image_index]["height"])

        flat_annotation_data = annotation["segmentation"][0]
        polygon_order = len(flat_annotation_data) / 2
        assert polygon_order.is_integer()

        vertices = np.array(np.array_split(flat_annotation_data, polygon_order), dtype=float)

        if invert_y_axis:
            x, y = vertices.T
            vertices = np.array([x, y_res - y]).T

        result[image_name][label] = init_polygon(
            vertices,
            label=label,
            reference_point_array=reference_point_array,
            frame=read_image_from_path(image_root / image_name) if image_root else None,
            manual_resolution=np.array((coco["images"][image_index]["width"], y_res), dtype=float),
            makesense_image_name=image_name,
            **current_kwargs,
            **perimeter_kwargs,
        )

    return perimeter_set_from_image_name_to_perimeters(dict(result))


def init_polygon_from_makesense_csv_rectangle(
    data_path: FilePath,
    image_root: Optional[DirectoryPath] = None,
    reference_point_csv_path: Optional[FilePath] = None,
    invert_y_axis: bool = runtime_settings.matplotlib_invert_y_axis,
    **perimeter_kwargs,
) -> dict[str, "PerimeterSet"]:
    from bikipy.perimeter.base import perimeter_set_from_image_name_to_perimeters
    from bikipy.perimeter.polygon.base import init_polygon

    logger.debug("Generating BasePolygonPerimeter from makesense polygon data in coco format")

    csv_data = read_makesense_rectangle(data_path, invert_y_axis)

    if reference_point_csv_path:
        image_name_to_reference_point = image_name_to_point_from_makesense(reference_point_csv_path)

    result = defaultdict(dict)
    for label, row in csv_data.iterrows():
        start = np.array(row[:2], dtype=int)
        end = start + np.array(row[2:4], dtype=int)

        image_name = row[5]

        if "reference_point_array" not in perimeter_kwargs:
            perimeter_kwargs["reference_point_array"] = (
                image_name_to_reference_point[image_name] if reference_point_csv_path else None
            )

        image_name = row["image_name"]
        result[image_name][label] = init_polygon(
            np.array((start, (start[0], end[1]), end, (end[0], start[1])), dtype=float),
            label=label,
            manual_resolution=np.array((row["x_res"], row["y_res"]), dtype=float),
            frame=read_image_from_path(image_root / str(image_name)) if image_root else None,
            makesense_image_name=row["image_name"],
            **perimeter_kwargs,
        )

    return perimeter_set_from_image_name_to_perimeters(dict(result))
