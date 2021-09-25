from pathlib import PurePath
from typing import Union

import numpy as np
import pandas as pd

from bikipy.perimeter import TriangularPerimeter, ParallelogramPerimeter
from bikipy.perimeter.base import PolygonalPerimeter


def generate_radial_arm_maze_arm_perimeters(
    line_csv_path: Union[PurePath, str],
    center_coco_path: Union[PurePath, str, None] = None,
    triangular_center_object: Union[TriangularPerimeter, None] = None,
):
    center_object = triangular_center_object or PolygonalPerimeter.from_coco(
        center_coco_path, single_obj_return=True
    )
    if not center_object:
        msg = "Either center_object or center_coco_path has to be defined"
        raise ValueError(msg)

    csv_array = pd.read_csv(
        line_csv_path,
        header=None,
        # names=["x1", "y1", "x2", "y2", "filename", "img_x", "img_y"],
    ).to_numpy()
    labels = tuple(csv_array.T[0])
    number_of_arms = len(labels)

    line_dataset = csv_array.T[1:5].T.astype(np.float)
    lines = [np.array_split(line, 2) for line in line_dataset]

    line_midpoints = [np.mean(lines[i], axis=0) for i in range(number_of_arms)]
    midpoint_scalars = np.linalg.norm(line_midpoints, axis=1)

    paired, arm_perimeters = [], []
    for line_index, midpoint_scalar in enumerate(midpoint_scalars):
        line_pair_index = np.where(
            np.argsort(midpoint_scalar - center_object.edge_midpoint_scalars) == 0
        )[0][0]
        assert line_pair_index not in paired
        paired.append(line_pair_index)
        arm_perimeters.append(
            ParallelogramPerimeter(
                np.concatenate((center_object.linked_corners[line_pair_index:line_pair_index+2], lines[line_index])),
                semantic_label=labels[line_index]
            )
        )
    print(paired)


if __name__ == "__main__":
    generate_radial_arm_maze_arm_perimeters(
        line_csv_path="/home/can/Software_Projects/BiKiPy/examples/ymaze_behaj_analysis/area_images/phd/A/coco_line_labels.csv",
        center_coco_path="/home/can/Software_Projects/BiKiPy/examples/ymaze_behaj_analysis/area_images/phd/A/coco_triangle.json",
    )
