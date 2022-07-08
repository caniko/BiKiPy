from functools import cached_property
from glob import iglob

import numpy as np
import pandas as pd

from bikipy.core.typing import TrialId
from bikipy.ingress.plugin.base import BasePluginDirectory, HasReferenceMixin
from bikipy.perimeter.base import SinglePerimeter, PerimeterSet
from bikipy.perimeter.polygon.makesense import init_polygon_from_makesense_coco_polygon
from bikipy.perimeter.polygon.rectangle import RectanglePerimeter
from bikipy.utils.collection_utils import get_first_value_in_dict
from bikipy.utils.makesense import read_makesense_line
from bikipy.utils.math.geometry import clockwise_argsort_points, meter_per_pixel_from_diagonal


class PluginRadial(BasePluginDirectory, HasReferenceMixin):
    ingress_key = "radial_definition_strategy"
    code_key = "radial"
    bikipy_trial_key = "radial"
    human_readable_index = "Radial"

    _center: SinglePerimeter | None = None

    @property
    def label(self):
        return self._info[1]

    @cached_property
    def line_data(self) -> pd.DataFrame:
        return read_makesense_line(self.data_path / "arm-lines.csv")

    @property
    def center(self) -> SinglePerimeter:
        if not self._center:
            self._center = get_first_value_in_dict(
                init_polygon_from_makesense_coco_polygon(
                    next(iglob(str(self.data_path / "center*"))),
                    reference_point_array=self.reference_point,
                    group_label="center",
                    inspect_arg=self.ingress.inspect_directory_path,
                )
            ).get_only_perimeter

        return self._center

    @cached_property
    def arms(self) -> list[RectanglePerimeter]:
        lines = np.array([np.array_split(line, 2) for _, line in self.line_data.iloc[:, 1:5].iterrows()])
        line_midpoints = np.mean(lines, axis=1)

        correct_argsort = clockwise_argsort_points(line_midpoints)

        lines = lines[correct_argsort]
        line_midpoints = line_midpoints[correct_argsort]

        arm_perimeters = []
        for line_index, line_midpoint in enumerate(line_midpoints):
            line_pair_bool_index = np.where(
                (
                    np.argsort(
                        np.linalg.norm(line_midpoint[None, :] - self.center.pixel_graph.vertex_midpoints, axis=1)
                    )
                    == 0
                )
            )[0][0]

            arm_perimeter_vertices = np.concatenate(
                (
                    self.center.pixel_graph.vertex_pairs[line_pair_bool_index],
                    lines[line_index],
                )
            )
            index_data = self.line_data.loc[line_index, :]
            arm_perimeter = RectanglePerimeter(
                vertices_in_pixels=arm_perimeter_vertices,
                int_id=line_index + 1,
                label=index_data["label"],
                reference_point_array=self.reference_point,
                manual_recording_resolution=np.array((index_data["x_res"], index_data["y_res"]), dtype=float),
                group_label="arms",
                inspect_arg=self.ingress.inspect_directory_path,
            )
            arm_perimeter.meters_per_pixel = meter_per_pixel_from_diagonal(
                arm_perimeter.vertices_in_pixels[0],
                arm_perimeter.vertices_in_pixels[2],
                self.ingress.settings["perimeter"]["radial_arm_rectangle_diagonal"],
            )
            # arm_perimeter.plot_perimeter(with_midpoints=True)
            arm_perimeters.append(arm_perimeter)

        arm_perimeter_mean_meters_per_pixel = PerimeterSet(perimeters=arm_perimeters).mean_meters_per_pixel
        for arm_perimeter in arm_perimeters:
            arm_perimeter.meters_per_pixel = arm_perimeter_mean_meters_per_pixel

        self._center.meters_per_pixel = arm_perimeter_mean_meters_per_pixel

        return arm_perimeters

    @cached_property
    def grouped_radial_maze_perimeters(self) -> dict[str, tuple[SinglePerimeter, ...]]:
        perimeter_set = PerimeterSet(perimeters=[*self.arms, self.center])

        grouped = perimeter_set.group()
        self.ingress.ingress_defined_perimeters[self.label] = grouped

        return grouped

    def trialwise_and_metadata(self, trial_id: TrialId) -> dict[str, tuple[SinglePerimeter, ...]]:
        return self.grouped_radial_maze_perimeters

    @property
    def globally_defined(self) -> dict[str, tuple[SinglePerimeter, ...]]:
        return self.grouped_radial_maze_perimeters
