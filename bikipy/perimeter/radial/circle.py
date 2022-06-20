from functools import cached_property
from typing import Any, Optional

import numpy as np
from matplotlib import pyplot as plt
from pydantic import FilePath, validator

from bikipy.core.typing import NDArrayFp64
from bikipy.core.video import VideoMetadata
from bikipy.perimeter.base import BasePerimeter
from bikipy.perimeter.radial.utils import plot_circle
from bikipy.utils.io.makesense import (
    get_line_endpoints_from_makesense_row,
    read_makesense_line,
)
from bikipy.utils.math.vector import unit_vector


class CirclePerimeter(BasePerimeter):
    center_in_pixels: NDArrayFp64
    radius: float

    @cached_property
    def corners(self):
        return self.center_in_pixels * self.video.meters_per_pixel

    @validator("center_in_pixels")
    def center_vector_is_2d(cls, value):
        if value.shape == (2,):
            pass
        elif value.shape == (1, 2):
            value = value[0]
        elif value.shape == (2, 1):
            value = value.T[0]
        else:
            msg = f"The center of {cls.__name__} must be a single coordinate tuple"
            raise ValueError(msg)
        return value.astype(float)

    @classmethod
    def from_makesense_line(cls, data_path: FilePath, **perimeter_kwargs) -> dict[str, Any]:
        result = {}
        for _, row in read_makesense_line(data_path).iterrows():
            a, b = get_line_endpoints_from_makesense_row(row)

            perimeter = cls(
                center_in_pixels=a,
                radius=np.linalg.norm(a - b),
                label=row["label"],
                manual_recording_resolution=np.array((row["x_res"], row["y_res"]), dtype=float),
                **perimeter_kwargs,
            )

            if row["image_name"] not in result:
                result[row["image_name"]] = {}
            result[row["image_name"]][row["label"]] = perimeter

        return cls.perimeter_set_from_image_name_to_perimeters(result)

    def change_reference(self, new_reference: NDArrayFp64, **new_inspect_image_kwargs):
        kwargs = self.dict()
        kwargs["center"] += new_reference - self.reference_point_array
        kwargs["reference_point_array"] = new_reference
        return self._new_inspect_image(self.__class__(**kwargs), **new_inspect_image_kwargs)

    def plot_perimeter(
        self,
        perimeter_border_normal_pixel_magnitude: Optional[float],
        ax: Any = None,
        include_geometric_legend: bool = False,
        colormap: Any = None,
    ):
        return plot_circle(self.center, self.radius, ax if ax else plt.subplots()[1])

    def coordinate_confinement_boolean_index(self, coordinates: NDArrayFp64, *args, **kwargs):
        distance_of_point_from_center = np.linalg.norm(coordinates - self.center, axis=1)
        return np.abs(distance_of_point_from_center) <= self.radius

    def expand(self, perimeter_border_normal_pixel_magnitude: float | NDArrayFp64):
        kwargs = self.dict()
        kwargs["radius"] += perimeter_border_normal_pixel_magnitude
        return self.__class__(**kwargs)

    def closest_sides_to_coordinates(self, coordinates: NDArrayFp64):
        return unit_vector(coordinates - self.center)
