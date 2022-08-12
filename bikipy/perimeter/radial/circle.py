from functools import cached_property
from typing import Any, Optional

import numpy as np
from matplotlib import pyplot as plt
from pydantic import FilePath, validator

from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64
from bikipy.feature.attention.gaze import gaze_direction_filter_circle_triangle
from bikipy.perimeter.base import (
    BaseSinglePerimeter,
    perimeter_set_from_image_name_to_perimeters,
)
from bikipy.perimeter.radial.utils import plot_circle
from bikipy.utils.makesense import (
    get_line_endpoints_from_makesense_row,
    read_makesense_line,
    recording_resolution_from_makesense_row,
)
from bikipy.utils.math.inside.ellipse import point_inside_ellipse
from bikipy.utils.math.vector import unit_vector
from bikipy.utils.plotting import generic_inspection_finalization


class CirclePerimeter(BaseSinglePerimeter):
    center_pixels: NDArrayFp64
    radius_pixels: float

    @classmethod
    @property
    def exclude_from_settings_schema(cls) -> set[str]:
        """
        Some required fields for a class are sometimes highly specific to its respective object. These fields should
        be recorded in this class-property to be excluded by the settings generator function in the ingress module
        :return:
        """
        return super().exclude_from_settings_schema.union({"center_pixels", "radius_pixels"})

    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.append(self.center_pixels.data.tobytes())
        result.append(self.radius_pixels)
        return result

    @property
    def centroid_meters(self) -> NDArrayFp64:
        return self.center_meters

    @cached_property
    def center_meters(self) -> NDArrayFp64:
        return self.center_pixels * self.video.meters_per_pixel

    @cached_property
    def radius_meters(self) -> float:
        return np.mean(self.radius_pixels * self.video.meters_per_pixel)

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

    def expand(self, perimeter_border_normal_pixels: float | NDArrayFp64):
        kwargs = self.dict()
        kwargs["radius_pixels"] += perimeter_border_normal_pixels
        return self.__class__(**kwargs)

    def compute_confined_coordinate_boolean_index(self, coordinates: NDArrayFp64, *args, **kwargs):
        if isinstance(self.radius_meters, float):
            distance_of_point_from_center = np.linalg.norm(coordinates - self.center_meters, axis=1)
            return np.abs(distance_of_point_from_center) <= self.radius_meters
        elif isinstance(self.radius_meters, np.ndarray):
            return point_inside_ellipse(coordinates, self.center_meters, self.radius_meters)

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

    @cached_property
    def scaled_center_in_pixels(self) -> NDArrayFp64:
        """
        Vertices must be scaled in accordance with video frame multiplier.
        :return:
        """
        if self.video.image_resize_multiplier:
            return self.center_pixels * self.video.image_resize_multiplier
        return self.center_pixels

    @cached_property
    def scaled_radius_in_pixels(self) -> float:
        """
        Vertices must be scaled in accordance with video frame multiplier.
        :return:
        """
        if self.video.image_resize_multiplier:
            return self.radius_pixels * self.video.image_resize_multiplier
        return self.radius_pixels

    def plot_perimeter(
        self,
        inspect_pixels: bool = False,
        manual_ax: Any = None,
        **plot_kwargs,
    ):
        if not manual_ax:
            fig, ax = plt.subplots()
        else:
            ax = manual_ax

        ax = (
            plot_circle(self.scaled_center_in_pixels, self.scaled_radius_in_pixels, ax)
            if inspect_pixels
            else plot_circle(self.center_meters, self.radius_meters, ax)
        )

        if not manual_ax:
            generic_inspection_finalization(self.class_inspect_arg or True, f"{self.label}.jpg")

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
                radius_pixels=np.linalg.norm((a - b)),  # AB vector is in pixels, must be meters
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
