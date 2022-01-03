import json
from functools import lru_cache
from logging import getLogger
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import DirectoryPath, FilePath

from bikipy.perimeter.base import Perimeter
from bikipy.perimeter.utils import reference_point_from_coco_path

logger = getLogger(__name__)


def from_makesense_coco_polygon(
    metadata_path: Any,
    image_root: Optional[DirectoryPath] = None,
    reference_point_csv_path: Optional[FilePath] = None,
    single_obj_return: bool = False,
    **perimeter_kwargs,
):
    logger.debug("Generating Perimeter from makesense polygon data in coco format")

    with open(metadata_path, "rb") as in_json:
        coco = json.load(in_json)

    assert not image_root or (image_root := Path(image_root)).exists()

    # The coco annotations are not sorted with respect to the category IDs
    coco["annotations"] = sorted(
        coco["annotations"], key=lambda dictionary: dictionary["category_id"]
    )

    # We don't need to do this, but better to be on the safe side
    coco["categories"] = sorted(
        coco["categories"], key=lambda dictionary: dictionary["id"]
    )

    if reference_point_csv_path:
        reference_data = reference_point_from_coco_path(
            reference_point_csv_path, single_row=False
        )
        assert len(reference_data) == len(
            coco["annotations"]
        ), f"{len(reference_data)} != {len(coco['annotations'])}"

    semantic_label_vs_polygon = {}
    for annotation in coco["annotations"]:
        current_kwargs = {}
        image_name = coco["images"][annotation["image_id"] - 1]["file_name"]
        if image_root:
            assert not any(
                key in perimeter_kwargs
                for key in ("inspect_image_path", "inspect_image_array")
            )
            current_kwargs["inspect_image_path"] = image_root / image_name
        if reference_point_csv_path:
            current_kwargs["reference_point_array"] = reference_data[image_name]

        semantic_label_vs_polygon[annotation["category_id"]] = Perimeter.init_polygon(
            _coco_polygon_annotation(annotation["segmentation"][0]),
            label=coco["categories"][annotation["category_id"] - 1]["name"],
            **current_kwargs,
            **perimeter_kwargs,
        )

    if single_obj_return:
        assert (
            len(semantic_label_vs_polygon) == 1
        ), f"More than one item in coco set, {len(semantic_label_vs_polygon)}"
        return semantic_label_vs_polygon.popitem()[1]

    return semantic_label_vs_polygon


def from_makesense_csv_rectangle(
    metadata_path: FilePath,
    image_root: DirectoryPath,
    reference_point_csv_path: Optional[FilePath] = None,
    **perimeter_kwargs,
):
    csv_data = pd.read_csv(metadata_path, header=None, index_col=0)

    if reference_point_csv_path:
        reference_data = reference_point_from_coco_path(
            reference_point_csv_path, single_row=False
        )
        assert len(reference_data) == len(
            csv_data
        ), f"{len(reference_data)} != {len(csv_data)}"

    result = []
    for label, row in csv_data.iterrows():
        current_kwargs = {}

        image_name = row[4]
        if reference_point_csv_path:
            current_kwargs["reference_point_array"] = reference_data[image_name]

        start = np.array(row[:2])
        end = start + np.array(row[2:4])

        result.append(
            cls.init_polygon(
                (start, (start[0], end[1]), end, (end[0], start[1])),
                inspect_image_path=image_root / image_name if image_root else None,
                label=label,
                **current_kwargs,
                **perimeter_kwargs,
            )
        )
    return result


def many_references_from_single_reference_file(
    reference_path: FilePath,
    image_root: DirectoryPath,
    **kwargs,
):
    pass


def _coco_polygon_annotation(flat_annotation_data: Sequence):
    return [
        (flat_annotation_data[i], flat_annotation_data[i + 1])
        for i in range(0, len(flat_annotation_data) - 1, 2)
    ]
