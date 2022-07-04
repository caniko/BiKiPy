from functools import cached_property
from glob import iglob

import numpy as np
import pandas as pd
from pydantic import PositiveInt

from bikipy.ingress.plugin.base import BasePluginDirectory, HasReferenceMixin, MetadataSupportError
from bikipy.perimeter.base import AnyPerimeter, PerimeterSet
from bikipy.perimeter.polygon.makesense import init_polygon_from_makesense_coco_polygon
from bikipy.perimeter.polygon.rectangle import RectanglePerimeter
from bikipy.utils.collection_utils import get_first_value_in_dict
from bikipy.utils.io.makesense import read_makesense_line
from bikipy.utils.math.geometry import clockwise_argsort_points


class PluginRadial(BasePluginDirectory, HasReferenceMixin):
    ingress_key = "radial_definition_strategy"
    code_key = "radial"
    bikipy_trial_key = "perimeter_set"
    human_readable_index = "Radial"

    @property
    def label(self):
        return self._info[1]

    @cached_property
    def line_data(self) -> pd.DataFrame:
        return read_makesense_line(self.data_path / "arm-lines.csv")

    @cached_property
    def center(self) -> AnyPerimeter:
        return get_first_value_in_dict(
            init_polygon_from_makesense_coco_polygon(next(iglob(str(self.data_path / "center*"))))
        )

    @cached_property
    def arms(self) -> list[RectanglePerimeter]:
        lines = np.array([np.array_split(line, 2) for line in self.line_data.iloc[1:5].T])
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

        return arm_perimeters

    @cached_property
    def radial_maze_perimeter_set(self) -> PerimeterSet:
        return PerimeterSet(perimeters=[*self.arms, self.center])

    def trialwise(self, trial_id: str | PositiveInt) -> PerimeterSet:
        return self.radial_maze_perimeter_set

    def metadata(self, key: str) -> PerimeterSet:
        raise MetadataSupportError(self.__class__.__name__)

    @property
    def globally_defined(self) -> PerimeterSet:
        return self.radial_maze_perimeter_set
