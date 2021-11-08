import os.path
from typing import Union

import numpy as np
from matplotlib import pyplot as plt

from bikipy.math.geometry import argsort_counterclockwise
from bikipy.perimeter import ParallelogramPerimeter, TriangularPerimeter
from bikipy.perimeter.base import Perimeter, PerimeterSet
from bikipy.utils.misc import read_makesense_point_csv
from bikipy.utils.typing import OptionalPathTyping, PathTyping


def generate_radial_arm_maze_arm_perimeters(
    line_csv_path: PathTyping,
    center_coco_path: OptionalPathTyping = None,
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
        center_object = triangular_center_object or Perimeter.from_coco(
            center_coco_path, single_obj_return=True
        )
    else:
        msg = "Either center_object or center_coco_path has to be defined"
        raise ValueError(msg)

    center_object.label = "center"
    center_object.group_label = "center"

    csv_array = read_makesense_point_csv(line_csv_path).T
    labels = csv_array[0]
    number_of_arms = len(labels)
    center_object.int_id = number_of_arms + 1

    line_dataset = csv_array[1:5].T.astype(np.float32)
    lines = np.array([np.array_split(line, 2) for line in line_dataset])
    line_midpoints = np.array([np.mean(line, axis=0) for line in lines])

    correct_argsort = argsort_counterclockwise(line_midpoints)
    lines = lines[correct_argsort]
    line_midpoints = line_midpoints[correct_argsort]

    arm_perimeters = []
    for line_index, line_midpoint in enumerate(line_midpoints):
        line_pair_index = np.where(
            np.argsort(
                np.linalg.norm(line_midpoint - center_object.edge_midpoints, axis=1)
            )
            == 0
        )[0][0]

        arm_perimeter = np.concatenate(
            (
                center_object.corners[
                    center_object.linked_polygon_edge_corner_pairs[line_pair_index], :
                ],
                lines[line_index],
            )
        )

        arm_perimeters.append(
            ParallelogramPerimeter(
                arm_perimeter,
                int_id=line_index + 1,
                label=labels[line_index],
                group_label="arm",
            )
        )

    perimeters = (*arm_perimeters, center_object)
    if inspect:
        fig, ax = plt.subplots(ncols=3)
        Perimeter.plot_perimeters(perimeters, ax=ax[0])
        for i, (line, center_corner) in enumerate(
            zip(lines, center_object.corners), start=1
        ):
            ax[1].scatter(*line.T, label=f"line_{i}")
            ax[2].scatter(*center_corner.T, label=f"center_corners_{i}")
        plt.legend()
        plt.show()

    return PerimeterSet(perimeters, **perimeter_kwargs)
