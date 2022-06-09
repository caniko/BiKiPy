from typing import Any, Optional

import numpy as np
from matplotlib import pyplot as plt
from pydantic import FilePath, validator
from pydantic_numpy import NDArray

from bikipy.perimeter.base import BasePerimeter
from bikipy.perimeter.radial.utils import plot_circle
from bikipy.utils.io.makesense import read_makesense_line, get_line_endpoints_from_makesense_row
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
    def read_makesense_line(cls, data_path: FilePath) -> dict[str, Any]:
        result = {}
        for _, row in read_makesense_line(data_path).iterrows():
            a, b = get_line_endpoints_from_makesense_row(row)
            perimeter = cls(center=a, radius=np.linalg.norm(a - b))

            if row["image_name"] not in result:
                result["image_name"] = {}
            result["image_name"][row["label"]] = perimeter

        return cls._perimeter_set_from_image_name_to_perimeters(result)

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
