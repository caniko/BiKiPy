from functools import cached_property
from glob import iglob

import numpy as np
import pandas as pd
from pydantic import computed_field
from pydantic_numpy import NpNDArrayFp64

from bikipy import runtime_settings
from bikipy.core.typing import Label
from bikipy.ingress.name_parser import PluginFileStemParseLastIsLabel
from bikipy.ingress.plugin.core.base import BasePluginDirectory
from bikipy.ingress.plugin.core.mixins import HasReferenceMixin, IngressRequiredMixin
from bikipy.math.geometry import clockwise_argsort_points, meter_per_pixel_from_diagonal
from bikipy.perimeter.base import PerimeterSet, BaseSinglePerimeter
from bikipy.perimeter.polygon.makesense import init_polygon_from_makesense_coco_polygon
from bikipy.perimeter.polygon.rectangle import RectanglePerimeter
from bikipy.utils.collection_utils import get_first_value_in_dict
from bikipy.utils.makesense import (
    get_all_lines_from_makesense_line_df,
    read_makesense_line,
)


class PluginRadial(BasePluginDirectory, HasReferenceMixin, IngressRequiredMixin):
    radial_arm_rectangle_diagonal: float
    derive_meters_per_pixel: bool = True

    plugin_file_stem_parser = PluginFileStemParseLastIsLabel

    ingress_key = "radial"
    code_key = "radial"
    default_trial_argument_key = "radial"
    human_readable_index = "Radial"

    _center: BaseSinglePerimeter | None = None

    @computed_field  # type: ignore[misc]
    @cached_property
    def line_data(self) -> pd.DataFrame:
        raw_line_data = read_makesense_line(
            self.data_path / "arm-lines.csv", invert_y_axis=runtime_settings.matplotlib_invert_y_axis
        )

        lines = get_all_lines_from_makesense_line_df(raw_line_data)
        line_midpoints = np.mean(lines, axis=1)
        clockwise_argsort = clockwise_argsort_points(line_midpoints)

        return raw_line_data.iloc[clockwise_argsort, :]

    @computed_field  # type: ignore[misc]
    @property
    def lines(self) -> NpNDArrayFp64:
        return get_all_lines_from_makesense_line_df(self.line_data)

    @computed_field  # type: ignore[misc]
    @property
    def center(self) -> BaseSinglePerimeter:
        if not self._center:
            self._center = get_first_value_in_dict(
                init_polygon_from_makesense_coco_polygon(
                    next(iglob(str(self.data_path / "center*"))),
                    reference_point_array=self.reference_point,
                    group_label="center",
                    derived_meters_per_pixel_source="side",
                    inspection_fig_output_path=self.ingress.inspect_directory_path,
                )
            ).get_only_perimeter

        return self._center

    @computed_field  # type: ignore[misc]
    @cached_property
    def arms(self) -> list[RectanglePerimeter]:
        center_vertex_pair = np.array(
            [self.center.pixel_graph.vertex_pairs[-1], *self.center.pixel_graph.vertex_pairs[:-1]]
        )
        arm_perimeters = []
        for line_index in range(len(self.lines)):
            arm_perimeter_vertices = np.concatenate(
                (
                    center_vertex_pair[line_index],
                    self.lines[line_index],
                )
            )
            index_data = self.line_data.loc[line_index, :]
            arm_perimeter = RectanglePerimeter(
                vertices_in_pixels=arm_perimeter_vertices,
                int_id=line_index + 1,
                label=index_data["label"],
                reference_point_array=self.reference_point,
                recording_resolution=np.array((index_data["x_res"], index_data["y_res"]), dtype=float),
                group_label="arms",
                inspection_fig_output_path=self.ingress.inspect_directory_path,
            )
            arm_perimeter.meters_per_pixel = meter_per_pixel_from_diagonal(
                arm_perimeter.vertices_in_pixels[0],
                arm_perimeter.vertices_in_pixels[2],
                self.radial_arm_rectangle_diagonal,
            )
            arm_perimeters.append(arm_perimeter)

        arm_perimeter_mean_meters_per_pixel = PerimeterSet(perimeters=arm_perimeters).mean_meters_per_pixel
        for arm_perimeter in arm_perimeters:
            arm_perimeter.meters_per_pixel = arm_perimeter_mean_meters_per_pixel

        self._center.meters_per_pixel = arm_perimeter_mean_meters_per_pixel

        return arm_perimeters

    @computed_field  # type: ignore[misc]
    @cached_property
    def grouped_radial_maze_perimeters(self) -> dict[str, tuple[BaseSinglePerimeter, ...]]:
        perimeter_set = PerimeterSet(perimeters=[*self.arms, self.center])

        # perimeter_set.plot(with_midpoints=True)

        grouped = perimeter_set.group()

        self.ingress.ingress_defined_perimeters[self.stem_info.label] = grouped
        return grouped

    def trialwise_and_metadata(self, trial_id: Label, naive: bool = False) -> dict[str, tuple[BaseSinglePerimeter, ...]]:
        self._assert_correct_scope_trialwise_metadata()
        return self.grouped_radial_maze_perimeters

    @computed_field  # type: ignore[misc]
    @property
    def globally_defined(self) -> dict[str, tuple[BaseSinglePerimeter, ...]]:
        self._assert_correct_scope_global()
        return self.grouped_radial_maze_perimeters
