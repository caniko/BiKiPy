from typing import Any, Optional

import numpy as np
from matplotlib import pyplot as plt
from pydantic import FilePath, validator
from pydantic_numpy import NDArray

from bikipy.perimeter.base import BasePerimeter
from bikipy.perimeter.radial.utils import plot_circle
from bikipy.utils.io.makesense import from_makesense_line
from bikipy.utils.math.vector import unit_vector


class CirclePerimeter(BasePerimeter):
    center: NDArray
    radius: float

    @validator("center")
    def center_vector_is_2d(cls, value):
        if value.shape == (2,):
            return value
        elif value.shape == (1, 2):
            return value[0]
        elif value.shape == (2, 1):
            return value.T[0]
        else:
            msg = f"The center of {cls.__name__} must be a single coordinate tuple"
            raise ValueError(msg)

    @classmethod
    def from_makesense_line(cls, data_path: FilePath) -> dict[str, Any]:
        return {
            label: cls(center=segment_tip_a, radius=np.linalg.norm(segment_tip_a - segment_tip_b))
            for label, (segment_tip_a, segment_tip_b) in from_makesense_line(data_path).items()
        }

    def change_reference(self, new_reference: NDArray, **new_inspect_image_kwargs):
        kwargs = self.dict()
        kwargs["center"] += new_reference - self.reference_point_array
        kwargs["reference_point_array"] = new_reference
        return self._new_inspect_image(self.__class__(**kwargs), **new_inspect_image_kwargs)

    def plot_perimeter(
        self,
        additional: Optional[float] = None,
        ax: Any = None,
        include_geometric_legend: bool = False,
        colormap: Any = None,
    ):
        return plot_circle(self.radius, ax if ax else plt.subplots()[1])

    def coordinate_confinement_boolean_index(self, coordinates: NDArray, *args, **kwargs):
        distance_of_point_from_center = np.linalg.norm(coordinates - self.center, axis=0)
        return np.abs(distance_of_point_from_center) <= self.radius

    def expand(self, additional: float):
        kwargs = self.dict()
        kwargs["radius"] += additional
        return self.__class__(**kwargs)

    def closest_sides_to_coordinates(self, coordinates: NDArray):
        line_unit_vector = unit_vector(coordinates - self.center)
        return self.center + line_unit_vector * self.radius
