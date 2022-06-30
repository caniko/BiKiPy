from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import PositiveInt

from bikipy.ingress.plugin.base import BasePluginDirectory
from bikipy.ingress.plugin.perimeter import PluginPerimeter
from bikipy.perimeter.base import AnyPerimeter, PerimeterSet
from bikipy.perimeter.polygon.rectangle import RectanglePerimeter
from bikipy.perimeter.utils import plot_perimeters
from bikipy.utils.io.makesense import read_makesense_line
from bikipy.utils.math.geometry import clockwise_argsort_points


class PluginRadial(BasePluginDirectory):
    trial_id: str | PositiveInt
    ingress: Any

    @property
    def label(self):
        return self._info[1]

    @cached_property
    def _input_perimeters(self) -> list[PluginPerimeter]:
        return [
            PluginPerimeter(data_path=perimeter_path, trial_id=self.trial_id, ingress=self.ingress)
            for perimeter_path in self.data_path.glob("perimeter*")
        ]

    @cached_property
    def center(self) -> AnyPerimeter:
        center_perimeter_paths = tuple(self.data_path.glob("*center*"))
        assert len(center_perimeter_paths) == 1
        return PluginPerimeter(
            data_path=center_perimeter_paths[0], ingress=self.ingress, trial_id=self.trial_id
        ).get_only_perimeter

    @cached_property
    def line_data(self) -> pd.DataFrame:
        line_data = [read_makesense_line(data_path) for data_path in self.data_path.glob("*line*")]
        line_data = line_data[0] if len(line_data) == 1 else pd.concat(line_data, axis=0)
        return line_data

    @cached_property
    def radial_maze_perimeters(self):
        line_dataset = self.line_data[1:5].T

        lines = np.array([np.array_split(line, 2) for line in line_dataset])
        line_midpoints = np.array([np.mean(line, axis=0) for line in lines])

        correct_argsort = clockwise_argsort_points(line_midpoints)
        lines = lines[correct_argsort]
        line_midpoints = line_midpoints[correct_argsort]

        arm_perimeters = []
        for line_index, line_midpoint in enumerate(line_midpoints):
            line_pair_index = np.where(
                np.argsort(np.linalg.norm(line_midpoint - self.center.edge_midpoints, axis=1)) == 0
            )[0][0]

            arm_perimeter = np.concatenate(
                (
                    self.center.vertices_in_meters[self.center.linked_polygon_edge_corner_pairs[line_pair_index], :],
                    lines[line_index],
                )
            )

            arm_perimeters.append(
                RectanglePerimeter(
                    vertices_in_pixels=arm_perimeter,
                    int_id=line_index + 1,
                    label=self.line_data["labels"][line_index],
                    group_label="arms",
                )
            )

        perimeters = [*arm_perimeters, self.center]

        if self._inspect:
            fig, ax = plt.subplots(ncols=3)
            plot_perimeters(perimeters, ax=ax[0])
            for i, (line, center_corner) in enumerate(zip(lines, self.center.vertices_in_meters), start=1):
                ax[1].scatter(*line.T, label=f"line_{i}")
                ax[2].scatter(*center_corner.T, label=f"center_vertices_{i}")
            plt.legend()
            plt.show()

        return PerimeterSet(perimeters=perimeters)
