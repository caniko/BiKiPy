from pathlib import PurePath
from typing import Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from bikipy.perimeter import ParallelogramPerimeter, TriangularPerimeter
from bikipy.perimeter.base import PolygonalPerimeter


def generate_radial_arm_maze_arm_perimeters(
    line_csv_path: Union[PurePath, str],
    center_coco_path: Union[PurePath, str, None] = None,
    triangular_center_object: Union[TriangularPerimeter, None] = None,
    inspect_image: Union[PurePath, None] = None,
    inspect: bool = False,
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

    line_dataset = csv_array.T[1:5].T.astype(np.float32)

    lines = [np.array_split(line, 2) for line in line_dataset]
    line_midpoints = [np.mean(lines[i], axis=0) for i in (2, 1, 0)]

    if inspect:
        fig, ax = plt.subplots(number_of_arms)
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
                semantic_label=labels[line_index],
                inspect_image=inspect_image,
            )
        )

    return arm_perimeters


if __name__ == "__main__":
    from pathlib import Path

    ROOT = (
        Path(
            "C:\\Users\\Can\\Projects\\BiKiPy"
            # "/home/can/Software_Projects/BiKiPy"
        )
        / "examples"
        / "ymaze_behaj_analysis"
        / "area_images"
        / "phd"
        / "A"
    )

    img = ROOT / "after_1_phd.png"
    a = generate_radial_arm_maze_arm_perimeters(
        line_csv_path=ROOT / "coco_line_labels.csv",
        center_coco_path=ROOT / "coco_triangle.json",
        inspect_image=img,
    )
    PolygonalPerimeter.plot_perimeters(a)
    # a[0].plot_parallelogram_labels()
    plt.show()
