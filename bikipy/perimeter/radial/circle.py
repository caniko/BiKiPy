from functools import cached_property
from typing import Any, Optional

import numpy as np
from matplotlib import pyplot as plt
from pydantic import FilePath, validator

from bikipy.core.typing import NDArrayBool, NDArrayFp64
from bikipy.feature.attention.gaze import gaze_direction_filter_circle_triangle
from bikipy.perimeter.base import (
    BaseSinglePerimeter,
    perimeter_set_from_image_name_to_perimeters,
)
from bikipy.perimeter.radial.utils import plot_circle
from bikipy.utils.io.makesense import (
    get_line_endpoints_from_makesense_row,
    read_makesense_line,
    recording_resolution_from_makesense_row,
)
from bikipy.utils.math.vector import unit_vector


class CirclePerimeter(BaseSinglePerimeter):
    center_pixels: NDArrayFp64
    radius_meters: float

    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.append(self.radius_meters)
        result.append(self.center_pixels.data.tobytes())
        return result

    @cached_property
    def center_meters(self):
        return self.center_pixels * self.video.meters_per_pixel

    @cached_property
    def radius_pixels(self):
        return self.radius_meters * self.video.pixels_per_meter

    @validator("center_pixels")
    def center_vector_is_2d(cls, value):
        if value.shape == (2,):
            pass
        elif value.shape == (1, 2):
            value = value[0]
        elif value.shape == (2, 1):
            value = value.T[0]
        else:
            msg = f"The center_meters of {cls.__name__} must be a single coordinate tuple"
            raise ValueError(msg)
        return value.astype(float)

    def change_reference(self, new_reference: NDArrayFp64, makesense_image_name: Optional[str] = None):
        return self.copy(
            update={
                "center_meters": self.center_meters + new_reference - self.reference_point_array,
                "reference_point_array": new_reference,
                "makesense_image_name": makesense_image_name,
            }
        )

    def expand(self, perimeter_border_normal_meters: float | NDArrayFp64):
        kwargs = self.dict()
        kwargs["radius_meters"] += perimeter_border_normal_meters
        return self.__class__(**kwargs)

    def confined_coordinate_boolean_index(self, coordinates: NDArrayFp64, *args, **kwargs):
        distance_of_point_from_center = np.linalg.norm(coordinates - self.center_meters, axis=1)
        return np.abs(distance_of_point_from_center) <= self.radius_meters

    def closest_point_on_edge_to_coordinates(self, coordinates: NDArrayFp64) -> NDArrayFp64:
        return self.center_meters + self.radius_meters * unit_vector(self.vector_to_closest_point_on_edge(coordinates))

    def vector_to_closest_point_on_edge(self, coordinates: NDArrayFp64) -> NDArrayFp64:
        """
        Strictly for circles, these vectors are the closest normals from the circle
        :param coordinates:
        :return:
        """
        return unit_vector(self.center_meters - coordinates)

    def gaze_direction_filter(self, *args, **kwargs) -> NDArrayBool:
        return gaze_direction_filter_circle_triangle(self, *args, **kwargs)

    def plot_perimeter(
        self,
        inspect_pixels: bool = False,
        perimeter_border_normal_pixels: Optional[float] = None,
        manual_ax: Any = None,
        **plot_kwargs,
    ):
        if not manual_ax:
            fig, ax = plt.subplots()
        else:
            ax = manual_ax

        if inspect_pixels:
            ax = plot_circle(self.center_pixels, self.radius_pixels, ax)
        else:
            ax = plot_circle(self.center_meters, self.radius_meters, ax)

        if not manual_ax:
            plt.show()
        if self.inspect_directory:
            plt.savefig(self.class_inspect_directory / f"{self.label}.jpg")

        return ax

    @classmethod
    def from_makesense_line(
        cls, data_path: FilePath, meters_per_pixel: NDArrayFp64, **perimeter_kwargs
    ) -> dict[str, Any]:
        result = {}
        for _, row in read_makesense_line(data_path).iterrows():
            a, b = get_line_endpoints_from_makesense_row(row)

            perimeter = cls(
                center_pixels=a,
                radius_meters=np.linalg.norm((a - b) * meters_per_pixel),  # AB vector is in pixels, must be meters
                label=row["label"],
                manual_recording_resolution=recording_resolution_from_makesense_row(row),
                makesense_image_name=row["image_name"],
                meters_per_pixel=meters_per_pixel,
                **perimeter_kwargs,
            )

            if row["image_name"] not in result:
                result[row["image_name"]] = {}
            result[row["image_name"]][row["label"]] = perimeter

        return perimeter_set_from_image_name_to_perimeters(result)
