from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property, partial, reduce
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Optional, Type, TypeVar

import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib.axes import Axes
from numpy import unsignedinteger
from pydantic import (
    DirectoryPath,
    Field,
    FilePath,
    computed_field,
    model_validator,
    validate_call,
)
from pydantic_numpy.typing import (
    NpNDArrayBool,
    NpNDArrayFp64,
    NpNDArrayInt16,
    NpNDArrayUint8,
)

from bikipy._constant import INSPECT_SIMPLE_FIG_FILE_FORMAT
from bikipy.core.base import BikipyHashable
from bikipy.core.mixin import InspectPlotMixin
from bikipy.core.typing import Label
from bikipy.core.video import VideoMetadata, VideoMetadataMixin
from bikipy.perimeter.polygon.makesense import (
    init_polygon_from_makesense_coco_polygon,
    init_polygon_from_makesense_csv_rectangle,
)
from bikipy.perimeter.utils.misc import get_coco_array_from_path_or_array
from bikipy.utils.makesense import get_point_from_makesense_row, read_makesense_point
from bikipy.utils.plot.generic import (
    ax_plot_coordinate_with_boolean_index,
    plot_coordinates,
)
from bikipy.utils.plot.inspect import generic_inspection_finalization

if TYPE_CHECKING:
    from bikipy.reader.base import Reader

logger = getLogger(__name__)

StringPerimeterShapes = Literal["circle", "circle_line", "circle_point", "polygon", "rectangle"]


class BasePerimeter(BikipyHashable, InspectPlotMixin, ABC):
    category = "perimeter"

    perimeter_label: ClassVar[str]

    def post_confinement_analysis_inspect_plot(
        self,
        boolean_index: NpNDArrayBool,
        coordinates: Optional[NpNDArrayFp64] = None,
        ax: Axes = None,
        inspection_fig_output_path: Path | None = None,
        **inspect_kwargs,
    ):
        if not inspection_fig_output_path:
            return

        if ax is None:
            fig, ax = self.video.subplot()

        self.plot_perimeter_on_ax(ax)

        if coordinates is not None:
            coordinates = self.video.prepare_coordinates_for_plotting(coordinates)
            ax_plot_coordinate_with_boolean_index(ax, boolean_index, coordinates)

        generic_inspection_finalization(
            inspection_fig_output_path,
            potential_dir=f"{self.perimeter_label}_{self.label}_confinement",
            inspect_fig_file_format=INSPECT_SIMPLE_FIG_FILE_FORMAT,
            **inspect_kwargs,
        )

    def subplot(self, manual_video: Optional[VideoMetadata] = None, **plot_kwargs):
        return manual_video.subplots(**plot_kwargs) if manual_video else self.video.subplots(**plot_kwargs)

    def plot_perimeter(self, manual_video: Optional[VideoMetadata] = None, manual_ax=None, **plot_kwargs):
        if manual_ax:
            ax = manual_ax
        else:
            fig, ax = (manual_video or self.video).subplot()

        self.plot_perimeter_on_ax(ax, **plot_kwargs)

    @abstractmethod
    def compute_confinement_boolean_index(
        self,
        coordinates: NpNDArrayFp64,
        manual_video: Optional[VideoMetadata] = None,
        ax: Axes = None,
        **inspect_kwargs,
    ) -> NpNDArrayBool:
        ...

    @abstractmethod
    def plot_perimeter_on_ax(
        self,
        ax: Axes,
        coordinates_as_pixels: bool = False,
        with_resize: bool = True,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        **plot_kwargs,
    ) -> None:
        ...

    @abstractmethod
    def change_reference(self, new_reference: NpNDArrayFp64, makesense_image_name: Optional[str] = None):
        ...

    @property
    @abstractmethod
    def centroid_meters(self) -> NpNDArrayFp64:
        ...


PerimeterCLS = Type[BasePerimeter]
Perimeter = TypeVar("Perimeter", bound=BasePerimeter)


from bikipy.reader.base import BaseReader  # ruff ignore E402

BaseReader.model_rebuild()


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
    derived_meters_per_pixel_source: Optional[str]
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

    required_video_metadata_fields = {"recording_resolution"}

    @computed_field(return_type=set[str])  # type: ignore[misc]
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
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.append(self.int_id)
        return result

    @computed_field(return_type=VideoMetadata)  # type: ignore[misc]
    @property
    def _video(self) -> VideoMetadata:
        upstream_video = super()._video

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

    @computed_field  # type: ignore[misc]
    @property
    def derived_meters_per_pixel(self) -> float | None:
        return

    @abstractmethod
    def expand(self, perimeter_border_normal_meters: float | NpNDArrayFp64):
        ...

    @abstractmethod
    def closest_point_on_edge_to_coordinates(self, coordinates: NpNDArrayFp64) -> NpNDArrayFp64:
        ...

    @abstractmethod
    def vector_to_closest_point_on_edge(self, coordinates: NpNDArrayFp64) -> NpNDArrayFp64:
        ...

    @abstractmethod
    def ray_direction_filter(
        self, ray_start_point: NpNDArrayFp64, ray_travel_direction_point: NpNDArrayFp64, max_radians: float, **kwargs
    ) -> NpNDArrayBool:
        ...

    @model_validator(mode="before")
    def mutually_exclusive(cls, values):
        if all(key in values and values[key] for key in ("reference_point_coco_path", "reference_point_array")):
            msg = "reference_point_coco_path and reference_point_array must be " "defined mutually exclusive"
            raise AttributeError(msg)
        return values

    def confinement_coordinate_boolean_index(
        self, coordinates: NpNDArrayFp64, reader: Optional["Reader"] = None, **inspect_kwargs
    ) -> NpNDArrayBool:
        """
        This function integrates moving perimeter routine into the static perimeter workflow
        """
        if not self.moving_field_name:
            return self.compute_confinement_boolean_index(coordinates, **inspect_kwargs)
        if not reader:
            msg = "reader must be passed to Perimeter when moving field name is utilized"
            raise AttributeError(msg)

        # if self.moving_field_name == "reference_point_array":

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
        coco_array: Optional[NpNDArrayFp64],
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
        coco_array: Optional[NpNDArrayFp64],
        image_root: Optional[DirectoryPath],
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

    def _method_name_inspect_arg(self, method_name: str) -> str:
        if self.makesense_image_name:
            return f"{self.makesense_image_name.split('.')[0]}-{method_name}"
        return method_name

    def plot(
        self,
        ax: Axes = None,
        coordinates: Optional[NpNDArrayFp64] = None,
        coordinates_as_pixels: bool = False,
        **perimeter_plot_kwargs,
    ):
        """
        Plot the perimeter using matplotlib. Optionally, plot coordinates alongside the perimeter

        Parameters
        ----------
        ax
            Axes object that the plot will be saved in. A new instance of Axes will be used
            if object returns False.
        coordinates
            Sequence of 2D coordinates that will be plotted alongside the perimeter
        perimeter_plot_kwargs

        Returns
        -------
        Axes object with plots
        """
        if not ax:
            fig, ax = self.video.subplot()
            ax.set_title(self.label)

        if coordinates is not None:
            plot_coordinates(ax, coordinates, coordinates_as_pixels, self.video)

        ax.set_title(self.label)

        self.plot_perimeter_on_ax(ax, **perimeter_plot_kwargs)


SinglePerimeter = TypeVar("SinglePerimeter", bound=BaseSinglePerimeter)


# TODO: Variadic generics Pydantic V2.1
class PerimeterSet(BasePerimeter):
    perimeters: list[SinglePerimeter]
    restricting_perimeters: Optional[list[SinglePerimeter]]

    @computed_field(return_type=set[str])  # type: ignore[misc]
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

    def __mod__(self, other: "PerimeterSet") -> "PerimeterSet":
        return PerimeterSet(
            perimeters=self.perimeters + other.perimeters,
            restricting_perimeters=self.restricting_perimeters + other.restricting_perimeters,
        )

    def __add__(self, other: SinglePerimeter) -> "PerimeterSet":
        # Subtraction includes the area in the PerimeterSet
        return PerimeterSet(
            perimeters=self.perimeters + other,
            restricting_perimeters=self.restricting_perimeters,
        )

    def __sub__(self, other: SinglePerimeter) -> "PerimeterSet":
        # Subtraction excludes the area from the PerimeterSet
        return PerimeterSet(
            perimeters=self.perimeters,
            restricting_perimeters=self.restricting_perimeters + other,
        )

    def __getitem__(self, item: Label) -> SinglePerimeter:
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
            return self.video.meters_per_pixel
        join_func = partial(VideoMetadata.join, meters_per_pixel_mean=True, ignore_incongruity=True)
        return reduce(join_func, (perimeter.video for perimeter in self.perimeters)).meters_per_pixel

    def group(self, pop_single_element_groups: bool = True) -> dict[str, tuple[SinglePerimeter, ...] | SinglePerimeter]:
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
    def centroid_meters(self) -> NpNDArrayFp64:
        """
        :return: The mean of all perimeter centroids in the set
        """
        return np.mean([perimeter.centroid_meters for perimeter in self.all_perimeters], axis=0)

    def combined_framewise_confinement_coordinates(self, coordinates: NpNDArrayFp64) -> NpNDArrayBool:
        present = np.any([perimeter.confinement_coordinate_boolean_index(coordinates) for perimeter in self.perimeters])
        if self.restricting_perimeters:
            present = present & ~np.any(
                [
                    perimeter.confinement_coordinate_boolean_index(coordinates)
                    for perimeter in self.restricting_perimeters
                ]
            )
        return present

    def compute_confinement_boolean_index(
        self,
        coordinates: NpNDArrayFp64,
        manual_video: Optional[VideoMetadata] = None,
        ax: Axes = None,
        **inspect_kwargs,
    ) -> NpNDArrayBool:
        result = self.combined_framewise_confinement_coordinates(coordinates)

        self.post_confinement_analysis_inspect_plot(result, coordinates, ax, **inspect_kwargs)

        return result

    def change_reference(self, **perimeter_change_reference_kwargs) -> "PerimeterSet":
        return self.__class__(
            perimeters=tuple(
                perimeter.change_reference(**perimeter_change_reference_kwargs) for perimeter in self.perimeters
            ),
            restricting_perimeters=tuple(
                perimeter.change_reference(**perimeter_change_reference_kwargs)
                for perimeter in self.restricting_perimeters
            )
            if self.restricting_perimeters
            else None,
        )

    @computed_field  # type: ignore[misc]
    @property
    def perimeter_to_int_id(self) -> dict[Perimeter, int]:
        return {perimeter: perimeter.int_id for perimeter in self.all_perimeters}

    @computed_field  # type: ignore[misc]
    @property
    def int_id_to_perimeter(self) -> dict[int, Perimeter]:
        return {perimeter.int_id: perimeter for perimeter in self.all_perimeters}

    @computed_field  # type: ignore[misc]
    @property
    def label_to_perimeter(self) -> dict[str, Perimeter]:
        return {perimeter.label: perimeter for perimeter in self.all_perimeters}

    @computed_field  # type: ignore[misc]
    @property
    def int_id_to_label(self) -> dict[int, str]:
        return {perimeter.int_id: perimeter.label for perimeter in self.all_perimeters}

    @computed_field  # type: ignore[misc]
    @property
    def reference_point(self) -> NpNDArrayFp64:
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
    def inspect_image_path(self) -> Path:
        result = self.all_perimeters[0].inspect_image_path
        assert all(result == perimeter.inspect_image_path for perimeter in self.all_perimeters)
        return result

    @computed_field  # type: ignore[misc]
    @property
    def all_perimeters(self) -> tuple[SinglePerimeter, ...]:
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
    def size_respective_dtype(self) -> Type[unsignedinteger]:
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
    def get_only_perimeter(self) -> SinglePerimeter:
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
        coordinates: Optional[NpNDArrayFp64] = None,
        coordinates_as_pixels: bool = False,
        **perimeter_plot_kwargs,
    ):
        if manual_ax is None:
            fig, ax = self.video.subplot(constrained_layout=True)
        else:
            ax = manual_ax

        with sb.color_palette("cubehelix", n_colors=self.number_of_vertices):
            for perimeter in self.all_perimeters:
                perimeter.plot_perimeter_on_ax(ax, coordinates_as_pixels=coordinates_as_pixels, **perimeter_plot_kwargs)

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


def perimeter_set_from_image_name_to_perimeters(image_name_to_perimeters: dict[str, "SinglePerimeter"]):
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
