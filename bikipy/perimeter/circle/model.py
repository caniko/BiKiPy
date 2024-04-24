from abc import ABC
from functools import cached_property
from typing import Optional, Self

import numpy as np
from matplotlib.axes import Axes
from pydantic import computed_field, field_validator
from pydantic_numpy.typing import Np1DArrayBool, Np2DArrayFp64, NpNDArrayInt16

from bikipy._constant import QUIVER_KWARGS
from bikipy.math.cached import meters2pixels
from bikipy.math.confinement.ellipse import point_inside_ellipse
from bikipy.math.vector import ray_direction_filter_circle_triangle, unit_vector
from bikipy.perimeter.base import BaseSinglePerimeter
from bikipy.utils.plot.generic import boolean_index_colormap, plot_ellipse


class BaseCirclePerimeter(BaseSinglePerimeter, ABC):
    center_pixels: NpNDArrayInt16

    perimeter_label = "circle"

    @field_validator("center_pixels")
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

    @computed_field  # type: ignore[misc]
    @property
    def centroid_meters(self) -> Np2DArrayFp64:
        return self.center_meters

    @computed_field  # type: ignore[misc]
    @cached_property
    def center_meters(self) -> Np2DArrayFp64:
        return self.center_pixels * self.meters_per_pixel

    def change_reference(self, new_reference: Np2DArrayFp64, makesense_image_name: Optional[str] = None) -> Self:
        return self.copy(
            update={
                "center_meters": self.center_meters + new_reference - self.reference_point_array,
                "reference_point_array": new_reference,
                "makesense_image_name": makesense_image_name,
            }
        )

    def expand(self, perimeter_border_normal_pixels: float | Np2DArrayFp64) -> "CircleFixedRadiusPerimeter":
        return CircleFixedRadiusPerimeter(
            center_pixels=self.center_pixels,
            radius_length_pixels=self.radius_length_pixels + perimeter_border_normal_pixels,
            manual_video=self.video,
        )

    def closest_point_on_edge_to_coordinates(self, coordinates: Np2DArrayFp64) -> Np2DArrayFp64:
        circle_center_to_point_uv = unit_vector(coordinates - self.center_meters)
        result = self.center_meters + self.radius_length_meters * circle_center_to_point_uv
        if self.inspect_closest_point_on_edge:
            self.plot_closest_point_on_edge_to_coordinates(coordinates, result)
        return result

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

    def _compute_confinement_boolean_index(self, coordinates: Np2DArrayFp64) -> Np1DArrayBool:
        if isinstance(self.radius_length_meters, float):
            distance_of_point_from_center = np.linalg.norm(coordinates - self.center_meters, axis=1)
            result = np.abs(distance_of_point_from_center) <= self.radius_length_meters
        elif isinstance(self.radius_length_meters, np.ndarray):
            result = point_inside_ellipse(coordinates, self.center_meters, self.radius_length_meters)
        else:
            raise RuntimeError

        return result

    def ray_direction_filter(
        self,
        op_label: str,
        ray_start_points: Np2DArrayFp64,
        ray_travel_direction_points: Np2DArrayFp64,
        max_radians: float,
        extra_ax: Optional[Axes] = None,
    ) -> Np1DArrayBool:
        closest_points_on_edges = self.closest_point_on_edge_to_coordinates(ray_travel_direction_points)
        vector_to_closest_point_on_edge = self.vector_to_closest_point_on_edge(
            ray_travel_direction_points, closest_points_on_edges
        )

        result, travel_direction_rays, _angles = ray_direction_filter_circle_triangle(
            ray_travel_direction_points,
            ray_start_points,
            vector_to_closest_point_on_edge,
            max_radians,
        )

        if not self.is_inspecting and not extra_ax and not self.inspect_ray_direction_filter:
            return result

        ray_travel_direction_points = self.video.prepare_coordinates_for_plotting(
            ray_travel_direction_points, step=True
        )
        vector_to_closest_point_on_edge = self.video.prepare_coordinates_for_plotting(
            vector_to_closest_point_on_edge, step=True
        )
        reduced_result = result[self.video.plot_stepper]

        if extra_ax:
            self.plot_perimeter_on_ax(ax=extra_ax)
            self._plot_ray_direction_filter(
                extra_ax,
                reduced_result,
                ray_travel_direction_points,
                travel_direction_rays,
                vector_to_closest_point_on_edge,
            )

        if not self.inspect_ray_direction_filter:
            return result

        fig, ax = self.video.subplot()

        self.plot_perimeter_on_ax(ax=ax)
        self._plot_ray_direction_filter(
            ax,
            reduced_result,
            ray_travel_direction_points,
            travel_direction_rays,
            vector_to_closest_point_on_edge,
        )

        self.save_fig(
            "ray-direction-filter",
            op_label,
            base_filename=f"{self.perimeter_label}-{self.label}",
            fig=fig,
        )

        return result

    def _plot_ray_direction_filter(
        self,
        ax: Axes,
        filter_boolean_index: Np1DArrayBool,
        ray_travel_direction_points: Np2DArrayFp64,
        travel_direction_rays: Np2DArrayFp64,
        vectors_to_closest_point_on_edge: Np2DArrayFp64,
    ) -> None:
        for (
            color,
            ray_travel_direction_point,
            travel_direction_ray,
            vector_to_closest_point_on_edge,
        ) in zip(
            boolean_index_colormap(filter_boolean_index),
            ray_travel_direction_points,
            travel_direction_rays,
            vectors_to_closest_point_on_edge,
        ):
            ax.quiver(*ray_travel_direction_point, *travel_direction_ray, color=color, **QUIVER_KWARGS)
            ax.quiver(*ray_travel_direction_point, *vector_to_closest_point_on_edge, color=color, **QUIVER_KWARGS)

    @classmethod
    @property
    def fields_to_exclude_from_single_schema(cls) -> set[str]:
        result = super().fields_to_exclude_from_single_schema
        result.add("center_pixels")
        return result

    @computed_field  # type: ignore[misc]
    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.append(self.center_pixels.data.tobytes())
        return result


class CircleVariableRadiusPerimeter(BaseCirclePerimeter):
    radius_length_meters: float

    @computed_field  # type: ignore[misc]
    @property
    def radius_length_pixels(self) -> float:
        return meters2pixels(self.radius_length_meters, self.video.pixels_per_meter)

    @computed_field(repr=False)  # type: ignore[misc]
    @property
    def derived_meters_per_pixel(self) -> float | None:
        raise NotImplementedError

    @computed_field  # type: ignore[misc]
    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.append(self.radius_length_meters)
        return result


class CircleFixedRadiusPerimeter(BaseCirclePerimeter):
    radius_length_pixels: float

    @computed_field  # type: ignore[misc]
    @property
    def derived_meters_per_pixel(self) -> float | None:
        if self.derived_meters_per_pixel_source_metric_length is not None:
            return self.derived_meters_per_pixel_source_metric_length / self.radius_length_pixels

    @computed_field  # type: ignore[misc]
    @cached_property
    def radius_length_meters(self) -> float:
        return np.mean(self.radius_length_pixels * self.meters_per_pixel)

    @classmethod
    @property
    def fields_to_exclude_from_single_schema(cls) -> set[str]:
        result = super().fields_to_exclude_from_single_schema
        result.add("radius_length_pixels")
        return result

    @computed_field  # type: ignore[misc]
    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.append(self.radius_length_pixels)
        return result


BaseCirclePerimeter.model_rebuild()
