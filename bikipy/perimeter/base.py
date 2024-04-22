from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property, partial, reduce
from logging import getLogger
from typing import ClassVar, Literal, Optional, Self

import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib.axes import Axes
from numpy import unsignedinteger
from pydantic import Field, FilePath, computed_field, model_validator, validate_call
from pydantic_numpy.typing import (
    Np1DArrayBool,
    Np2DArrayFp64,
    NpNDArrayInt16,
    NpNDArrayUint8,
)

from bikipy._constant import QUIVER_KWARGS
from bikipy.core.base import BikipyHashable
from bikipy.core.mixin import InspectPlotMixin
from bikipy.core.typing import Label
from bikipy.core.video import VideoMetadata, VideoMetadataMixin
from bikipy.math.vector import unit_vector
from bikipy.perimeter.polygon.makesense import (
    init_polygon_from_makesense_coco_polygon,
    init_polygon_from_makesense_csv_rectangle,
)
from bikipy.perimeter.utils.misc import get_coco_array_from_path_or_array
from bikipy.utils.makesense import get_point_from_makesense_row, read_makesense_point
from bikipy.utils.plot.generic import (
    ax_hue_plot_coordinate_pair_as_lines,
    ax_hue_plot_coordinate_with_boolean_index,
    ax_hue_plot_coordinates,
    color_map_by_number,
    plot_coordinates,
)

logger = getLogger(__name__)

StringPerimeterShapes = Literal["circle", "circle_line", "circle_point", "polygon", "rectangle"]


class BasePerimeter(BikipyHashable, InspectPlotMixin, ABC):
    category = "perimeter"

    perimeter_label: ClassVar[str]

    def confinement_boolean_index(
        self,
        op_label: str,
        boolean_index: Np1DArrayBool,
        coordinates: Np2DArrayFp64,
        extra_ax: Optional[Axes] = None,
    ):
        result = self._compute_confinement_boolean_index(coordinates)

        if not self.is_inspecting:
            return result

        fig, ax = self.video.subplot()

        coordinates = self.video.prepare_coordinates_for_plotting(coordinates)

        if extra_ax:
            ax_hue_plot_coordinate_with_boolean_index(extra_ax, boolean_index, coordinates)
            self.plot_perimeter_on_ax(ax=extra_ax)

        ax_hue_plot_coordinate_with_boolean_index(ax, boolean_index, coordinates)
        self.plot_perimeter_on_ax(ax=ax)

        self.save_fig(
            "perimeter-confinement",
            op_label,
            base_filename=f"{self.perimeter_label}-{self.label}",
            fig=fig,
        )

        return result

    @abstractmethod
    def ray_direction_filter(
        self,
        op_label: str,
        ray_start_points: Np2DArrayFp64,
        ray_travel_direction_points: Np2DArrayFp64,
        max_radians: float,
        extra_ax: Optional[Axes] = None,
    ) -> Np1DArrayBool: ...

    @abstractmethod
    def _compute_confinement_boolean_index(self, coordinates: Np2DArrayFp64) -> Np1DArrayBool: ...

    @abstractmethod
    def plot_perimeter_on_ax(
        self,
        ax: Axes,
        coordinates_as_pixels: bool = False,
        with_resize: bool = True,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        **plot_kwargs,
    ) -> None: ...

    @abstractmethod
    def change_reference(self, new_reference: Np2DArrayFp64, makesense_image_name: Optional[str] = None): ...

    @property
    @abstractmethod
    def centroid_meters(self) -> Np2DArrayFp64: ...


PerimeterCLS = type[BasePerimeter]


class BaseSinglePerimeter(BasePerimeter, VideoMetadataMixin, ABC):
    impenetrable: bool = Field(
        False,
        description="Signifies the impenetrability of the perimeter. "
        "Usually because the perimeter is insurmountable or slippery",
    )

    derive_meters_per_pixel: bool = Field(
        False,
        description="Derive meters per pixel from perimeter. The ratio is derived from source defined in "
        "meters_per_pixel_from_perimeter_source",
    )
    derived_meters_per_pixel_source: Optional[str] = None
    derived_meters_per_pixel_source_metric_length: Optional[float] = Field(
        None,
        description="Metric length of the pre-determined component, see derived pixels per pixel from perimeter in "
        "the documentation.",
    )

    int_id: Optional[int] = Field(None, description="For multi-perimeter trials where sequential confinement is used")
    group_label: Optional[str] = None

    makesense_image_name: Optional[str] = None

    reference_point_coco_path: Optional[FilePath] = None
    reference_point_array: Optional[NpNDArrayInt16] = None

    moving_field_name: Optional[str] = None

    required_video_metadata_fields = {"resolution"}

    @model_validator(mode="before")
    def mutually_exclusive(cls, values):
        if all(key in values and values[key] for key in ("reference_point_coco_path", "reference_point_array")):
            msg = "reference_point_coco_path and reference_point_array must be " "defined mutually exclusive"
            raise AttributeError(msg)
        return values

    @computed_field  # type: ignore[misc]
    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.append(self.int_id)
        return result

    def plot_closest_point_on_edge_to_coordinates(
        self, coordinates: Np2DArrayFp64, closest_point: Np2DArrayFp64, ax: Optional[Axes] = None
    ) -> Axes | None:
        if not self.is_inspecting:
            return

        if not ax:
            fig, ax = self.video.subplot()
        else:
            fig = None

        self.plot_perimeter_on_ax(ax=ax, coordinates_as_pixels=False)

        if self.video:
            coordinates = self.video.prepare_coordinates_for_plotting(coordinates, step=True)
            closest_point = self.video.prepare_coordinates_for_plotting(closest_point, step=True)

        ax_hue_plot_coordinates(ax, coordinates)
        ax_hue_plot_coordinate_pair_as_lines(ax, coordinates, closest_point)

        if not fig:
            return ax
        self.save_fig("closest_point_on_edge_to_coordinates", base_filename=self.label, fig=fig)

    def vector_to_closest_point_on_edge(
        self, coordinates: Np2DArrayFp64, closest_edge_points: Optional[Np2DArrayFp64] = None
    ) -> Np2DArrayFp64:
        """
        Strictly for circles, these vectors are the closest normals from the circle

        :param coordinates:
        :param closest_edge_points:
        :return:
        """
        if closest_edge_points is None:
            closest_edge_points = self.closest_point_on_edge_to_coordinates(coordinates)

        result = unit_vector(closest_edge_points - coordinates)

        self.plot_vector_to_closest_point_on_edge(coordinates, closest_edge_points, result)

        return result

    def plot_vector_to_closest_point_on_edge(
        self,
        coordinates: Np2DArrayFp64,
        closest_edge_points: Np2DArrayFp64,
        vectors: Np2DArrayFp64,
        ax: Optional[Axes] = None,
    ) -> Axes | None:
        if not self.is_inspecting:
            return

        if not ax:
            fig, ax = self.video.subplot()
        else:
            fig = None

        self.plot_perimeter_on_ax(ax, coordinates_as_pixels=False)

        coordinates = self.video.prepare_coordinates_for_plotting(coordinates, step=True)
        closest_edge_points = self.video.prepare_coordinates_for_plotting(closest_edge_points, step=True)
        vectors = self.video.prepare_coordinates_for_plotting(vectors[self.video.plot_stepper] * 8)

        for color, coord, closest_edge_point, vector in zip(
            color_map_by_number(len(coordinates)), coordinates, closest_edge_points, vectors
        ):
            ax.scatter(*closest_edge_point, color=color, marker="x")
            ax.quiver(*coord, *vector, color=color, **QUIVER_KWARGS)

        if not fig:
            return ax
        self.save_fig("vector_to_closest_point_on_edge", base_filename=self.label, fig=fig)

    @classmethod
    @property
    def fields_to_exclude_from_single_schema(cls) -> set[str]:
        result = super().fields_to_exclude_from_single_schema
        result.update(
            {
                "int_id",
                "group_label",
                "makesense_image_name",
                "reference_point_coco_path",
                "reference_point_array",
            }
        )
        return result

    @computed_field  # type: ignore[misc]
    @property
    def reference_point(self) -> pd.DataFrame | None:
        if self.reference_point_array is None and not self.reference_point_coco_path:
            return None
        return (
            read_makesense_point(self.reference_point_coco_path)
            if self.reference_point_array is None
            else self.reference_point_array
        )

    @reference_point.setter
    def reference_point(self, value) -> None:
        self.reference_point_array = np.ascontiguousarray(value)

    def change_reference_with_coco(
        self,
        metadata_path: Optional[FilePath],
        coco_array: Optional[Np2DArrayFp64],
        **kwargs,
    ):
        coco_array = get_coco_array_from_path_or_array(metadata_path, coco_array)

        if len(coco_array) != 1:
            msg = (
                "The coco array includes more than one annotation. "
                "Please use change_reference_with_coco_with_plural_references()"
            )
            raise ValueError(msg)

        return self.change_reference(get_point_from_makesense_row(coco_array), **kwargs)

    def change_reference_with_coco_with_plural_references(
        self,
        metadata_path: Optional[FilePath],
        coco_array: Optional[Np2DArrayFp64],
        map_to_image_names: bool = True,
    ):
        def _change_reference_loop_func(reference_point, img_name):
            return self.change_reference(
                reference_point,
            )

        coco_array = get_coco_array_from_path_or_array(metadata_path, coco_array)

        img_name_to_reference_points = {row[3]: get_point_from_makesense_row(row) for row in coco_array}
        if not np.any(self.reference_point):
            msg = "The reference polygon has no reference point"
            raise ValueError(msg)

        if map_to_image_names:
            return {
                img_name: _change_reference_loop_func(reference_point, img_name)
                for img_name, reference_point in img_name_to_reference_points.items()
            }
        return [
            _change_reference_loop_func(reference_point, img_name)
            for img_name, reference_point in img_name_to_reference_points.items()
        ]

    @computed_field(return_type=VideoMetadata)  # type: ignore[misc]
    @property
    def video(self) -> VideoMetadata:
        upstream_video = super().video

        if self.derive_meters_per_pixel:
            logger.debug("derive_meters_per_pixel -> True: Deriving meters_per_pixel from perimeter")
            if not self.derived_meters_per_pixel_source:
                msg = "Derivation source, meters_per_pixel_from_perimeter_source, for meters_per_pixel undefined"
                raise AttributeError(msg)
            if not self.derived_meters_per_pixel_source_metric_length:
                msg = (
                    "Length of source, length_meters_of_meters_per_pixel_source, "
                    "for deriving meters_per_pixel is undefined"
                )
                raise AttributeError(msg)

            upstream_video.meters_per_pixel = self.derived_meters_per_pixel

        return upstream_video

    @abstractmethod
    def expand(self, perimeter_border_normal_meters: float | Np2DArrayFp64): ...

    @abstractmethod
    def closest_point_on_edge_to_coordinates(self, coordinates: Np2DArrayFp64) -> Np2DArrayFp64: ...

    @property
    @abstractmethod
    def derived_meters_per_pixel(self) -> float | None: ...


class PerimeterSet(BasePerimeter):
    perimeters: list[BaseSinglePerimeter]
    restricting_perimeters: Optional[list[BaseSinglePerimeter]] = None

    @classmethod
    @property
    def fields_to_exclude_from_single_schema(cls) -> set[str]:
        result = super().fields_to_exclude_from_single_schema
        result.update(("perimeters", "restricting_perimeters"))
        return result

    @computed_field  # type: ignore[misc]
    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.extend(perimeter._to_hash for perimeter in self.all_perimeters)
        return result

    def __mod__(self, other: Self) -> Self:
        return PerimeterSet(
            perimeters=self.perimeters + other.perimeters,
            restricting_perimeters=self.restricting_perimeters + other.restricting_perimeters,
        )

    def __add__(self, other: BaseSinglePerimeter) -> Self:
        # Subtraction includes the area in the PerimeterSet
        return PerimeterSet(
            perimeters=self.perimeters + other,
            restricting_perimeters=self.restricting_perimeters,
        )

    def __sub__(self, other: BaseSinglePerimeter) -> Self:
        # Subtraction excludes the area from the PerimeterSet
        return PerimeterSet(
            perimeters=self.perimeters,
            restricting_perimeters=self.restricting_perimeters + other,
        )

    def __getitem__(self, item: Label) -> BaseSinglePerimeter:
        for perimeter in self.all_perimeters:
            if perimeter.label == item or perimeter.int_id == item:
                return perimeter
        raise KeyError(f"Item was not found, {item} amongst {self.all_perimeters}")

    @computed_field(return_type=VideoMetadata)  # type: ignore[misc]
    @cached_property
    def video(self) -> VideoMetadata:
        result = self.all_perimeters[0].video
        for p in self.all_perimeters[1:]:
            result = VideoMetadata.join(result, p.video)
        return result

    @computed_field  # type: ignore[misc]
    @cached_property
    def mean_meters_per_pixel(self) -> float:
        if self.number_of_perimeters == 1:
            return self.meters_per_pixel
        join_func = partial(VideoMetadata.join, meters_per_pixel_mean=True, ignore_incongruity=True)
        return reduce(join_func, (perimeter.video for perimeter in self.perimeters)).meters_per_pixel

    def group(
        self, pop_single_element_groups: bool = True
    ) -> dict[str, tuple[BaseSinglePerimeter, ...] | BaseSinglePerimeter]:
        grouped = defaultdict(list)
        for perimeter in self.all_perimeters:
            grouped[perimeter.group_label].append(perimeter)

        for label, perimeters in grouped.items():
            number_of_perimeters = len(perimeters)
            if number_of_perimeters > 1:
                grouped[label] = tuple(perimeters)
            elif number_of_perimeters == 1:
                grouped[label] = perimeters[0] if pop_single_element_groups else tuple(perimeters)

        return dict(grouped)

    @computed_field  # type: ignore[misc]
    @cached_property
    def centroid_meters(self) -> Np2DArrayFp64:
        """
        :return: The mean of all perimeter centroids in the set
        """
        return np.mean([perimeter.centroid_meters for perimeter in self.all_perimeters], axis=0)

    def _compute_confinement_boolean_index(self, coordinates: Np2DArrayFp64) -> Np1DArrayBool:
        result = np.any(
            [perimeter.confinement_boolean_index("framewise-confinement", coordinates) for perimeter in self.perimeters]
        )
        if self.restricting_perimeters:
            result = result & ~np.any(
                [
                    perimeter.confinement_boolean_index("restricted-framewise-confinement", coordinates)
                    for perimeter in self.restricting_perimeters
                ]
            )

        return result

    def _compute_ray_direction_filter(
        self, ray_start_points: Np2DArrayFp64, ray_travel_direction_points: Np2DArrayFp64, max_radians: float, **kwargs
    ) -> Np1DArrayBool:
        result = np.any(
            [
                perimeter.ray_direction_filter(ray_start_points, ray_travel_direction_points, max_radians)
                for perimeter in self.perimeters
            ]
        )
        if self.restricting_perimeters:
            result = result & ~np.any(
                [
                    perimeter.ray_direction_filter(ray_start_points, ray_travel_direction_points, max_radians)
                    for perimeter in self.restricting_perimeters
                ]
            )

        return result

    def _plot_ray_direction_filter(
        self, ax: Axes, ray_start_points: Np2DArrayFp64, ray_travel_direction_points: Np2DArrayFp64
    ) -> None:
        raise NotImplementedError("This method is not implemented for PerimeterSet")

    def change_reference(self, **perimeter_change_reference_kwargs) -> Self:
        return self.__class__(
            perimeters=tuple(
                perimeter.change_reference(**perimeter_change_reference_kwargs) for perimeter in self.perimeters
            ),
            restricting_perimeters=(
                tuple(
                    perimeter.change_reference(**perimeter_change_reference_kwargs)
                    for perimeter in self.restricting_perimeters
                )
                if self.restricting_perimeters
                else None
            ),
        )

    @computed_field  # type: ignore[misc]
    @property
    def perimeter_to_int_id(self) -> dict[BasePerimeter, int]:
        return {perimeter: perimeter.int_id for perimeter in self.all_perimeters}

    @computed_field  # type: ignore[misc]
    @property
    def int_id_to_perimeter(self) -> dict[int, BasePerimeter]:
        return {perimeter.int_id: perimeter for perimeter in self.all_perimeters}

    @computed_field  # type: ignore[misc]
    @property
    def label_to_perimeter(self) -> dict[str, BasePerimeter]:
        return {perimeter.label: perimeter for perimeter in self.all_perimeters}

    @computed_field  # type: ignore[misc]
    @property
    def int_id_to_label(self) -> dict[int, str]:
        return {perimeter.int_id: perimeter.label for perimeter in self.all_perimeters}

    @computed_field  # type: ignore[misc]
    @property
    def reference_point(self) -> Np2DArrayFp64:
        expected_reference_point = self.all_perimeters[0].reference_point
        if equality := np.all(
            expected_reference_point == perimeter.reference_point for perimeter in self.all_perimeters
        ):
            logger.warning("The reference points are different within the perimeter set")
        if not equality or not np.any(expected_reference_point):
            return None
        return expected_reference_point

    @computed_field  # type: ignore[misc]
    @property
    def inspect_image(self) -> NpNDArrayUint8:
        result = self.all_perimeters[0].inspect_image
        assert all(result == perimeter.inspect_image for perimeter in self.all_perimeters)
        return result

    @computed_field  # type: ignore[misc]
    @property
    def all_perimeters(self) -> tuple[BaseSinglePerimeter, ...]:
        if not self.restricting_perimeters:
            return tuple(self.perimeters)
        return *self.perimeters, *self.restricting_perimeters

    @computed_field  # type: ignore[misc]
    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(perimeter.label for perimeter in self.all_perimeters)

    @computed_field  # type: ignore[misc]
    @cached_property
    def number_of_perimeters(self) -> int:
        return len(self.all_perimeters)

    @computed_field  # type: ignore[misc]
    @cached_property
    def size_respective_dtype(self) -> type[unsignedinteger]:
        return np.uint8 if self.number_of_perimeters <= 255 else np.uint16

    @computed_field  # type: ignore[misc]
    @cached_property
    def number_of_vertices(self) -> int:
        vertex_numbers = []
        for p in self.all_perimeters:
            if hasattr(p, "polygon_order"):
                vertex_numbers.append(p.polygon_order)
            elif hasattr(p, "center_pixels"):  # circle
                vertex_numbers.append(1)
        return sum(vertex_numbers)

    @computed_field  # type: ignore[misc]
    @property
    def get_only_perimeter(self) -> BaseSinglePerimeter:
        assert self.number_of_perimeters == 1
        return self.all_perimeters[0]

    def plot_perimeter_on_ax(
        self,
        ax: Axes,
        coordinates_as_pixels: bool = False,
        with_resize: bool = True,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        **plot_kwargs,
    ) -> None:
        for perimeter in self.all_perimeters:
            perimeter.plot_perimeter_on_ax(
                ax, coordinates_as_pixels, with_resize, x_pixel_offset, y_pixel_offset, **plot_kwargs
            )

    def plot(
        self,
        manual_ax: Axes = None,
        coordinates: Optional[Np2DArrayFp64] = None,
        coordinates_as_pixels: bool = False,
        **perimeter_plot_kwargs,
    ):
        if manual_ax is None:
            fig, ax = self.video.subplot(constrained_layout=True)
        else:
            ax = manual_ax

        with sb.color_palette("cubehelix", n_colors=self.number_of_vertices):
            for perimeter in self.all_perimeters:
                perimeter.plot_perimeter_on_ax(
                    ax=ax, coordinates_as_pixels=coordinates_as_pixels, **perimeter_plot_kwargs
                )

            if coordinates is not None:
                plot_coordinates(ax, coordinates, coordinates_as_pixels, self.video)


@validate_call
def perimeter_set_from_makesense(
    perimeter_path: FilePath, shape: StringPerimeterShapes, **perimeter_kwargs
) -> dict[str, PerimeterSet]:
    unsupported_msg = f"Unsupported format, {shape}"

    match shape:
        case "circle_line" | "circle":
            from bikipy.perimeter.circle.makesense import circle_from_makesense_line

            return circle_from_makesense_line(perimeter_path, **perimeter_kwargs)
        case "circle_point":
            from bikipy.perimeter.circle.makesense import circle_from_makesense_point

            return circle_from_makesense_point(perimeter_path, **perimeter_kwargs)

        case "rectangle":
            match perimeter_path.suffix:
                case ".csv":
                    return init_polygon_from_makesense_csv_rectangle(perimeter_path, **perimeter_kwargs)
                case ".json":
                    return init_polygon_from_makesense_coco_polygon(perimeter_path, **perimeter_kwargs)
                case _:
                    raise ValueError(unsupported_msg)

        case "polygon" | "triangle":
            return init_polygon_from_makesense_coco_polygon(perimeter_path, **perimeter_kwargs)

        case _:
            raise ValueError(unsupported_msg)


def perimeter_set_from_image_name_to_perimeters(image_name_to_perimeters: dict[str, "BaseSinglePerimeter"]):
    result = {}
    for image_name, perimeters in image_name_to_perimeters.items():
        filtered_perimeters, restricting_perimeters = [], []
        for label, perimeter in perimeters.items():
            if isinstance(label, str) and label.lower().startswith("restricted"):
                restricting_perimeters.append(perimeter)
            else:
                filtered_perimeters.append(perimeter)
        result[image_name] = PerimeterSet(
            perimeters=filtered_perimeters, restricting_perimeters=restricting_perimeters, label=image_name
        )
    return result
