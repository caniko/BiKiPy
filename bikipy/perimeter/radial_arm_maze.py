import os.path
from typing import Union

import numpy as np
from matplotlib import pyplot as plt

from bikipy.math.geometry import argsort_counterclockwise
from bikipy.perimeter import ParallelogramPerimeter, TriangularPerimeter
from bikipy.perimeter.base import PolygonalPerimeter, PolygonalPerimeterSet
from bikipy.utils.misc import read_makesense_point_csv
from bikipy.utils.typing import Path_typing, Path_typing_kwarg


def generate_radial_arm_maze_arm_perimeters(
    line_csv_path: Path_typing,
    center_coco_path: Path_typing_kwarg = None,
    triangular_center_object: Union[TriangularPerimeter, None] = None,
    inspect: bool = False,
    **perimeter_kwargs,
):
    if triangular_center_object:
        center_object = triangular_center_object
    elif center_coco_path:
        if not os.path.exists(center_coco_path):
            msg = f"center_coco_path does not exist, {center_coco_path}"
            raise ValueError(msg)
        center_object = triangular_center_object or PolygonalPerimeter.from_coco(
            center_coco_path, single_obj_return=True
        )
    else:
        msg = "Either center_object or center_coco_path has to be defined"
        raise ValueError(msg)

    center_object.semantic_label = "center"
    center_object.group_label = "center"

    csv_array = read_makesense_point_csv(line_csv_path).T
    labels = csv_array[0]
    number_of_arms = len(labels)
    center_object.int_label = number_of_arms + 1

    line_dataset = csv_array[1:5].T.astype(np.float32)
    lines = np.array([np.array_split(line, 2) for line in line_dataset])
    line_midpoints = np.array([np.mean(line, axis=0) for line in lines])

    correct_argsort = argsort_counterclockwise(line_midpoints)
    lines = lines[correct_argsort]
    line_midpoints = line_midpoints[correct_argsort]

    if inspect:
        fig, ax = plt.subplots(len(labels))

    paired, arm_perimeters = [], []
    for line_index, line_midpoint in enumerate(line_midpoints):
        line_pair_index = np.where(
            np.argsort(
                np.linalg.norm(line_midpoint - center_object.edge_midpoints, axis=1)
            )
            == 0
        )[0][0]

        assert line_pair_index not in paired
        arm_perimeter = np.concatenate(
            (
                center_object.linked_corners[line_pair_index : line_pair_index + 2],
                lines[line_index],
            )
        )
        if inspect:
            ax[line_index].scatter(*arm_perimeter.T)
            ax[line_index] = center_object.plot_perimeter(ax=ax)

        paired.append(line_pair_index)
        arm_perimeters.append(
            ParallelogramPerimeter(
                arm_perimeter,
                int_label=line_index + 1,
                semantic_label=labels[line_index],
                group_label="arm",
            )
        )

    return PolygonalPerimeterSet((*arm_perimeters, center_object), **perimeter_kwargs)
