import os.path
from logging import getLogger
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from pydantic import FilePath

from bikipy.perimeter.base import PerimeterSet
from bikipy.perimeter.polygon.base import PolygonPerimeter
from bikipy.perimeter.polygon.makesense import init_polygon_from_makesense_coco_polygon
from bikipy.perimeter.polygon.rectangle import RectanglePerimeter
from bikipy.perimeter.polygon.triangular import TriangularPerimeter
from bikipy.utils.io.makesense import read_makesense_point
from bikipy.utils.math.geometry import clockwise_argsort_points

logger = getLogger(__name__)


def generate_radial_maze_perimeters(
    line_csv_path: FilePath,
    center_coco_path: Optional[FilePath],
    triangular_center_object: Optional[TriangularPerimeter],
    inspect: bool = False,
    **perimeter_kwargs,
):
    if triangular_center_object:
        center_object = triangular_center_object
    elif center_coco_path:
        if not os.path.exists(center_coco_path):
            msg = f"center_coco_path does not exist, {center_coco_path}"
            raise ValueError(msg)
        center_object = triangular_center_object or init_polygon_from_makesense_coco_polygon(
            center_coco_path, single_obj_return=True, **perimeter_kwargs
        )
    else:
        msg = "Either center_object or center_coco_path has to be defined"
        raise ValueError(msg)

    center_object.label = "center"
    center_object.group_label = "center"

    csv_array = read_makesense_point(line_csv_path).T
    labels = csv_array[0]
    number_of_arms = len(labels)
    center_object.int_id = number_of_arms + 1

    line_dataset = csv_array[1:5].T
    lines = np.array([np.array_split(line, 2) for line in line_dataset])
    line_midpoints = np.array([np.mean(line, axis=0) for line in lines])

    correct_argsort = clockwise_argsort_points(line_midpoints)
    lines = lines[correct_argsort]
    line_midpoints = line_midpoints[correct_argsort]

    arm_perimeters = []
    for line_index, line_midpoint in enumerate(line_midpoints):
        line_pair_index = np.where(
            np.argsort(np.linalg.norm(line_midpoint - center_object.line_segment_midpoints_meters, axis=1)) == 0
        )[0][0]

        arm_perimeter = np.concatenate(
            (
                center_object.vertices_in_meters[center_object.line_segment_points_pixels[line_pair_index], :],
                lines[line_index],
            )
        )

        arm_perimeters.append(
            RectanglePerimeter(
                vertices_in_pixels=arm_perimeter,
                int_id=line_index + 1,
                label=labels[line_index],
                group_label="arms",
                **perimeter_kwargs,
            )
        )

    perimeters = (*arm_perimeters, center_object)
    if inspect:
        fig, ax = plt.subplots(ncols=3)
        PolygonPerimeter.plot_perimeters(perimeters, ax=ax[0])
        for i, (line, center_corner) in enumerate(zip(lines, center_object.vertices_in_meters), start=1):
            ax[1].scatter(*line.T, label=f"line_{i}")
            ax[2].scatter(*center_corner.T, label=f"center_vertices_{i}")
        plt.legend()
        plt.show()

    return PerimeterSet(perimeters=perimeters)
