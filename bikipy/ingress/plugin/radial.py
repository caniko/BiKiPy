from functools import cached_property
from math import ceil
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import PositiveInt, DirectoryPath

from bikipy.ingress.plugin.base import BasePluginDirectory
from bikipy.ingress.plugin.perimeter import PluginPerimeter, PluginPerimeterMixin
from bikipy.perimeter.base import AnyPerimeter, PerimeterSet
from bikipy.perimeter.polygon.rectangle import RectanglePerimeter
from bikipy.utils.collection_utils import flatten_sequence, get_first_value_in_dict
from bikipy.utils.io.makesense import read_makesense_line
from bikipy.utils.math.geometry import clockwise_argsort_points


class PluginRadial(BasePluginDirectory, PluginPerimeterMixin):
    trial_id: str | PositiveInt
    ingress: Any

    @property
    def label(self):
        return self._info[1]

    @cached_property
    def _input_perimeters(self) -> list[PluginPerimeter]:
        return [
            PluginPerimeter(
                data_path=perimeter_path,
                trial_id=self.trial_id,
                ingress=self.ingress,
                manual_reference=self.reference_point,
            )
            for perimeter_path in self.data_path.glob("perimeter*")
        ]

    @cached_property
    def center_plugin_perimeter(self) -> PluginPerimeter:
        center_perimeter_paths = tuple(self.data_path.glob("*center*"))
        assert len(center_perimeter_paths) == 1
        return PluginPerimeter(data_path=center_perimeter_paths[0], ingress=self.ingress, trial_id=self.trial_id)

    @property
    def center(self) -> AnyPerimeter:
        return self.center_plugin_perimeter.get_only_perimeter

    @cached_property
    def line_data(self) -> pd.DataFrame:
        return read_makesense_line(next(self.data_path.glob("*line*")))

    @cached_property
    def radial_maze_perimeter_set(self) -> dict[str, PerimeterSet]:
        line_dataset = self.line_data.iloc[1:5].T

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
                    reference_point_array=self.reference_point,
                    group_label="arms",
                )
            )

        perimeters = [*arm_perimeters, self.center]

        annotated_perimeter_set = PerimeterSet(perimeters=perimeters)

        result = {self.center_plugin_perimeter.image_name: annotated_perimeter_set}
        for new_image_name, new_reference in self.image_name_to_re_referencing_point.items():
            result[new_image_name] = annotated_perimeter_set.change_reference(new_reference=new_reference)

        if self._inspect:
            fig, axes = plt.subplots(ncols=3, nrows=ceil(len(result) / 3.0), constrained_layout=True)
            axes = flatten_sequence(axes)

            for ax, (image_name, radial_maze_perimeter_set) in zip(axes, result.items()):
                radial_maze_perimeter_set.plot(ax=ax)
                ax.set_title(image_name if image_name != self.center_plugin_perimeter.image_name else "Source")

            plt.legend()
            plt.show()

        return result

    @property
    def get_only_perimeter_set(self) -> PerimeterSet:
        assert len(self.radial_maze_perimeter_set) == 1
        return get_first_value_in_dict(self.radial_maze_perimeter_set)


def radial_directory_path_to_value(
    file_path: DirectoryPath, trial_id: str | PositiveInt, ingress: Any, *args, **kwargs
):
    return PluginRadial(data_path=file_path, ingress=ingress, trial_id=trial_id).radial_maze_perimeter_set
