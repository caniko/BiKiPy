import json
from logging import getLogger
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from pydantic import DirectoryPath, FilePath

from bikipy.core.typing import NDArrayFp64
from bikipy.utils.io.makesense import (
    image_name_to_point_from_makesense,
    read_makesense_rectangle,
)

logger = getLogger(__name__)


def init_polygon_from_makesense_coco_polygon(
    data_path: FilePath,
    image_root: Optional[DirectoryPath] = None,
    reference_point_csv_path: Optional[FilePath] = None,
    manual_reference_point_array: Optional[NDArrayFp64] = None,
    invert_y: bool = False,
    **perimeter_kwargs,
) -> dict:
    from bikipy.perimeter.base import perimeter_set_from_image_name_to_perimeters
    from bikipy.perimeter.polygon.base import init_polygon

    logger.debug("Generating PolygonPerimeter from makesense polygon data in coco format")

    with open(data_path, "rb") as in_json:
        coco = json.load(in_json)

    assert not image_root or (image_root := Path(image_root)).exists()

    # The coco annotations are not sorted with respect to the category IDs
    coco["annotations"] = sorted(coco["annotations"], key=lambda dictionary: dictionary["category_id"])

    # We don't need to do this, but better to be on the safe side
    coco["categories"] = sorted(coco["categories"], key=lambda dictionary: dictionary["id"])

    if reference_point_csv_path:
        image_name_to_reference_point = image_name_to_point_from_makesense(reference_point_csv_path)

    result = {}
    for annotation in coco["annotations"]:
        current_kwargs = {}
        image_index = annotation["image_id"] - 1
        image_name = coco["images"][image_index]["file_name"]
        label = coco["categories"][annotation["category_id"] - 1]["name"]

        if image_name not in result:
            result[image_name] = {}

        if manual_reference_point_array is None:
            reference_point_array = image_name_to_reference_point[image_name] if reference_point_csv_path else None
        else:
            reference_point_array = manual_reference_point_array

        y_res = float(coco["images"][image_index]["height"])
        vertices = np.array(
            _coco_polygon_annotation(
                annotation["segmentation"][0], invert_y_vertical_resolution=coco["images"][image_index]["height"]
            )
        )
        if invert_y:
            x, y = vertices.T
            vertices = np.array([x, y_res - y]).T

        result[image_name]["label"] = init_polygon(
            vertices,
            label=label,
            reference_point_array=reference_point_array,
            manual_frame=cv2.imread(image_root / image_name) if image_root else None,
            manual_recording_resolution=np.array((coco["images"][image_index]["width"], y_res), dtype=float),
            makesense_image_name=image_name,
            **current_kwargs,
            **perimeter_kwargs,
        )

    return perimeter_set_from_image_name_to_perimeters(result)


def init_polygon_from_makesense_csv_rectangle(
    data_path: FilePath,
    image_root: Optional[DirectoryPath] = None,
    reference_point_csv_path: Optional[FilePath] = None,
    invert_y: bool = True,
    **perimeter_kwargs,
):
    from bikipy.perimeter.base import perimeter_set_from_image_name_to_perimeters
    from bikipy.perimeter.polygon.base import init_polygon

    logger.debug("Generating PolygonPerimeter from makesense polygon data in coco format")

    csv_data = read_makesense_rectangle(data_path)

    if reference_point_csv_path:
        image_name_to_reference_point = image_name_to_point_from_makesense(reference_point_csv_path)

    result = {}
    for label, row in csv_data.iterrows():
        start = np.array(row[:2], dtype=int)
        end = start + np.array(row[2:4], dtype=int)

        if "reference_point_array" not in perimeter_kwargs:
            perimeter_kwargs["reference_point_array"] = (
                image_name_to_reference_point[image_name] if reference_point_csv_path else None
            )

        image_name = row["image_name"]
        if image_name not in result:
            result[image_name] = {}

        y_res = float(row["y_res"])
        vertices = np.array((start, (start[0], end[1]), end, (end[0], start[1])))
        if invert_y:
            x, y = vertices.T
            vertices = np.array([x, y_res - y]).T

        result[image_name][label] = init_polygon(
            vertices,
            label=label,
            manual_recording_resolution=np.array((row["x_res"], y_res), dtype=float),
            manual_frame=cv2.imread(image_root / str(image_name)) if image_root else None,
            makesense_image_name=row["image_name"],
            **perimeter_kwargs,
        )

    return perimeter_set_from_image_name_to_perimeters(result)


def _coco_polygon_annotation(flat_annotation_data: Sequence, invert_y_vertical_resolution: Optional[float] = None):
    return [
        (
            flat_annotation_data[i],
            (
                invert_y_vertical_resolution - flat_annotation_data[i + 1]
                if invert_y_vertical_resolution
                else flat_annotation_data[i + 1]
            ),
        )
        for i in range(0, len(flat_annotation_data) - 1, 2)
    ]
