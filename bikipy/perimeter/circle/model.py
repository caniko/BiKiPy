from functools import cached_property
from typing import Optional, Type, TypeVar

import numpy as np
from matplotlib.axes import Axes
from pydantic import validator
from pydantic_numpy.dtype import NDArrayFp64, NDArrayInt16

from bikipy.core.video import VideoMetadata
from bikipy.perimeter.base import BaseSinglePerimeter
from bikipy.utils.math.cached import meters2pixels
from bikipy.utils.math.confinement.ellipse import point_inside_ellipse
from bikipy.utils.math.vector import unit_vector
from bikipy.utils.plot.generic import plot_ellipse


class BaseCirclePerimeter(BaseSinglePerimeter):
    center_pixels: NDArrayInt16

    perimeter_label = "circle"

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

    @property
    def centroid_meters(self) -> np.ndarray[float, np.dtype[np.float64]]:
        return self.center_meters

    @cached_property
    def center_meters(self) -> np.ndarray[float, np.dtype[np.float64]]:
        return self.center_pixels * self.video.meters_per_pixel

    def change_reference(self, new_reference: NDArrayFp64, makesense_image_name: Optional[str] = None):
        return self.copy(
            update={
                "center_meters": self.center_meters + new_reference - self.reference_point_array,
                "reference_point_array": new_reference,
                "makesense_image_name": makesense_image_name,
            }
        )

    def expand(self, perimeter_border_normal_pixels: float | NDArrayFp64) -> "BaseCirclePerimeter":
        kwargs = self.dict()
        kwargs["radius_length_pixels"] += perimeter_border_normal_pixels
        return self.__class__(**kwargs)

    def compute_confinement_boolean_index(
        self, coordinates: NDArrayFp64, manual_video: Optional[VideoMetadata] = None, ax: Axes = None, **inspect_kwargs
    ):
        if isinstance(self.radius_length_meters, float):
            distance_of_point_from_center = np.linalg.norm(coordinates - self.center_meters, axis=1)
            result = np.abs(distance_of_point_from_center) <= self.radius_length_meters
        elif isinstance(self.radius_length_meters, np.ndarray):
            result = point_inside_ellipse(coordinates, self.center_meters, self.radius_length_meters)
        else:
            raise RuntimeError

        self.post_confinement_analysis_inspect_plot(result, coordinates, manual_video, ax, **inspect_kwargs)

        return result

    def closest_point_on_edge_to_coordinates(self, coordinates: NDArrayFp64) -> np.ndarray[float, np.dtype[np.float64]]:
        return self.center_meters + self.radius_length_meters * unit_vector(
            self.vector_to_closest_point_on_edge(coordinates)
        )

    def vector_to_closest_point_on_edge(self, coordinates: NDArrayFp64) -> np.ndarray[float, np.dtype[np.float64]]:
        """
        Strictly for circles, these vectors are the closest normals from the circle
        :param coordinates:
        :return:
        """
        return unit_vector(self.center_meters - coordinates)

    def ray_direction_filter(
        self, ray_start_point: NDArrayFp64, ray_travel_direction_point: NDArrayFp64, max_radians: float, **kwargs
    ) -> np.ndarray[bool, bool]:
        from bikipy.behaviour.utils import ray_direction_filter_circle_triangle

        return ray_direction_filter_circle_triangle(self, ray_travel_direction_point, ray_start_point, max_radians)

    def plot_perimeter_on_ax(
        self,
        ax: Axes,
        coordinates_as_pixels: bool = False,
        with_resize: bool = True,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        **plot_kwargs,
    ) -> None:
        center, radius = (
            (self.center_pixels, self.radius_length_pixels)
            if coordinates_as_pixels
            else (self.center_meters, self.radius_length_meters)
        )

        center[0] += x_pixel_offset
        center[1] += y_pixel_offset

        if coordinates_as_pixels and with_resize:
            center *= self.video.image_resize_multiplier
            radius *= self.video.image_resize_multiplier

        if isinstance(radius, np.ndarray):
            radius = tuple(radius)

        plot_ellipse(ax, tuple(center), radius, **plot_kwargs)

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        result = super().schemantic_fields_to_exclude_from_config_schema
        result.add("center_pixels")
        return result

    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.append(self.center_pixels.data.tobytes())
        return result


CirclePerimeterCLS = Type[BaseCirclePerimeter]
CirclePerimeter = TypeVar("CirclePerimeter", bound=BaseCirclePerimeter)


class CircleVariableRadiusPerimeter(BaseCirclePerimeter):
    radius_length_meters: float

    @property
    def radius_length_pixels(self) -> float:
        return meters2pixels(self.radius_length_meters, self.video.pixels_per_meter)

    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.append(self.radius_length_meters)
        return result


class CircleFixedRadiusPerimeter(BaseCirclePerimeter):
    radius_length_pixels: float

    @property
    def derived_meters_per_pixel(self) -> float:
        return self.derived_meters_per_pixel_source_metric_length / self.radius_length_pixels

    @cached_property
    def radius_length_meters(self) -> float:
        return np.mean(self.radius_length_pixels * self.video.meters_per_pixel)

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        result = super().schemantic_fields_to_exclude_from_config_schema
        result.add("radius_length_pixels")
        return result

    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.append(self.radius_length_pixels)
        return result
