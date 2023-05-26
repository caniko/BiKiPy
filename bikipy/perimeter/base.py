from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cached_property, partial, reduce
from logging import getLogger
from typing import TYPE_CHECKING, Any, Literal, Optional, Type, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from pydantic import DirectoryPath, Field, FilePath, root_validator, validate_arguments
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64, NDArrayInt16

from bikipy._constant import INSPECT_FIG_FILE_FORMAT
from bikipy.core.base import BikipyHashable
from bikipy.core.mixin import InspectPlotMixin
from bikipy.core.typing import Label
from bikipy.core.video import VideoMetadata, VideoMetadataMixin
from bikipy.perimeter.helper.utils import get_coco_array_from_path_or_array
from bikipy.perimeter.polygon.makesense import (
    init_polygon_from_makesense_coco_polygon,
    init_polygon_from_makesense_csv_rectangle,
)
from bikipy.utils.makesense import get_point_from_makesense_row, read_makesense_point
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS
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
    def _manual_video_metadata_derived_inspection_preparation(
        self,
        manual_video: Optional[VideoMetadata] = None,
        coordinates: Optional[NDArrayFp64] = None,
        ax: Axes = None,
        **plot_kwargs,
    ) -> tuple[Any, NDArrayFp64, VideoMetadata]:
        if ax:
            if manual_video:
                msg = "manual_video and ax should defined mutually exclusively, contact developers please"
                raise ValueError(msg)
            return ax, coordinates, self.video

        _, ax = self.subplot(manual_video, **plot_kwargs)

        video: VideoMetadata = manual_video or self.video
        if coordinates is not None:
            coordinates: NDArrayFp64 = video.prepare_coordinates_for_plotting(coordinates)

        return ax, coordinates, video

    def _post_confinement_analysis_inspect_plot(
        self,
        boolean_index: NDArrayBool,
        coordinates: Optional[NDArrayFp64] = None,
        manual_video: Optional[VideoMetadata] = None,
        ax: Axes = None,
        **inspect_kwargs,
    ):
        if not self.inspect_arg:
            return

        ax, inspection_coordinates, video = self._manual_video_metadata_derived_inspection_preparation(
            manual_video, coordinates, ax
        )

        # _manual_video_metadata_derived_inspection_preparation -> video.subplot makes sure the axes is in a list,
        # we need to revert that action.
        if isinstance(ax, list):
            ax = ax[0]

        self.plot_perimeter_on_ax(
            ax,
            inspect_pixels=video.coordinates_need_to_be_scaled_for_plot,
            manual_resize_multiplier=video.image_resize_multiplier,
        )

        if coordinates is not None:
            ax_plot_coordinate_with_boolean_index(ax, boolean_index, inspection_coordinates)

        generic_inspection_finalization(self.class_inspect_arg, **inspect_kwargs)

    def subplot(self, manual_video: Optional[VideoMetadata] = None, **plot_kwargs):
        return manual_video.subplots(**plot_kwargs) if manual_video else self.video.subplots(**plot_kwargs)

    def plot_perimeter(
        self,
        manual_video: Optional[VideoMetadata] = None,
        manual_ax=None,
        **plot_kwargs,
    ):
        video = manual_video or self.video
        if manual_ax:
            ax = manual_ax
        else:
            fig, ax = video.subplot()

        return self.plot_perimeter_on_ax(
            ax, inspect_pixels=video.coordinates_need_to_be_scaled_for_plot, manual_video=video
        )

    @abstractmethod
    def compute_confined_coordinate_boolean_index(
        self, coordinates: NDArrayFp64, manual_video: Optional[VideoMetadata] = None, ax: Axes = None, **inspect_kwargs
    ) -> np.ndarray[bool, bool]:
        ...

    @abstractmethod
    def plot_perimeter_on_ax(
        self, ax: Axes, inspect_pixels: bool = False, manual_resize_multiplier: Optional[float] = None, **plot_kwargs
    ) -> Axes:
        ...

    @abstractmethod
    def change_reference(self, new_reference: NDArrayFp64, makesense_image_name: Optional[str] = None):
        ...

    @property
    @abstractmethod
    def centroid_meters(self) -> np.ndarray[float, np.float64]:
        ...


PerimeterCLS = Type[BasePerimeter]
Perimeter = TypeVar("Perimeter", bound=BasePerimeter)


from bikipy.reader.base import BaseReader

BaseReader.update_forward_refs(Perimeter=Perimeter)


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
        description="Metric length of the pre-determined component, see derived pixels per pixel from perimeter in "
        "the documentation.",
    )

    int_id: Optional[int] = Field(description="For multi-perimeter trials where sequential confinement is used")
    group_label: Optional[str]

    makesense_image_name: Optional[str]

    reference_point_coco_path: Optional[FilePath]
    reference_point_array: Optional[NDArrayInt16]

    moving_field_name: Optional[str]

    category = "perimeter"
    required_video_metadata_fields = {"recording_resolution"}

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(
            {
                "int_id",
                "group_label",
                "makesense_image_name",
                "reference_point_coco_path",
                "reference_point_array",
            }
        )
        return upstream

    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.append(self.int_id)
        return result

    @property
    def _video(self) -> VideoMetadata:
        upstream_video = super()._video

        if self.derive_meters_per_pixel:
            logger.debug("derive_meters_per_pixel -> True: Deriving meters_per_pixel from perimeter")
            if not self.derived_meters_per_pixel_source:
                msg = "Derivation source, meters_per_pixel_from_perimeter_source, for meters_per_pixel undefined"
                raise AttributeError(msg)
            if not self.derived_meters_per_pixel_source_metric_length:
                msg = "Length of source, length_meters_of_meters_per_pixel_source, for deriving meters_per_pixel is undefined"
                raise AttributeError(msg)

            upstream_video.meters_per_pixel = self.derived_meters_per_pixel

        return upstream_video

    @property
    def derived_meters_per_pixel(self) -> float | None:
        return

    @abstractmethod
    def expand(self, perimeter_border_normal_meters: float | NDArrayFp64) -> "SinglePerimeter":
        ...

    @abstractmethod
    def closest_point_on_edge_to_coordinates(self, coordinates: NDArrayFp64) -> np.ndarray[float, np.float64]:
        ...

    @abstractmethod
    def vector_to_closest_point_on_edge(self, coordinates: NDArrayFp64) -> np.ndarray[float, np.float64]:
        ...

    @abstractmethod
    def ray_direction_filter(
        self, ray_start_point: NDArrayFp64, ray_travel_direction_point: NDArrayFp64, max_radians: float, **kwargs
    ) -> np.ndarray[bool, bool]:
        ...

    @root_validator(pre=True)
    def mutually_exclusive(cls, values):
        if all(key in values and values[key] for key in ("reference_point_coco_path", "reference_point_array")):
            msg = "reference_point_coco_path and reference_point_array must be " "defined mutually exclusive"
            raise AttributeError(msg)
        return values

    def confined_coordinate_boolean_index(
        self, coordinates: NDArrayFp64, reader: Optional["Reader"] = None
    ) -> np.ndarray[bool, bool]:
        """
        This function integrates moving perimeter routine into the static perimeter workflow
        """
        if not self.moving_field_name:
            return self.compute_confined_coordinate_boolean_index(coordinates)
        if not reader:
            msg = "reader must be passed to Perimeter when moving field name is utilized"
            raise AttributeError(msg)

        # if self.moving_field_name == "reference_point_array":

    @property
    def reference_point(self):
        if self.reference_point_array is None and not self.reference_point_coco_path:
            return None
        return (
            read_makesense_point(self.reference_point_coco_path)
            if self.reference_point_array is None
            else self.reference_point_array
        )

    @reference_point.setter
    def reference_point(self, value):
        self.reference_point_array = np.ascontiguousarray(value)

    def change_reference_with_coco(
        self,
        metadata_path: Optional[FilePath],
        coco_array: Optional[NDArrayFp64],
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
        coco_array: Optional[NDArrayFp64],
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
        coordinates: Optional[NDArrayFp64] = None,
        inspect_pixels: bool = False,
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
            plot_coordinates(coordinates, ax, inspect_pixels, self.video)

        ax.set_title(self.label)

        return self.plot_perimeter(manual_ax=ax, inspect_pixels=inspect_pixels, **perimeter_plot_kwargs)


SinglePerimeter = TypeVar("SinglePerimeter", bound=BaseSinglePerimeter)


class PerimeterSet(BasePerimeter):
    perimeters: list[SinglePerimeter]
    restricted_perimeters: Optional[list[SinglePerimeter]]

    category = "PerimeterSet"

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("perimeters", "restricted_perimeters"))
        return upstream

    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.extend(perimeter._to_hash for perimeter in self.all_perimeters)
        return result

    def __mod__(self, other: "PerimeterSet") -> "PerimeterSet":
        return PerimeterSet(
            perimeters=self.perimeters + other.perimeters,
            restricted_perimeters=self.restricted_perimeters + other.restricted_perimeters,
        )

    def __add__(self, other: SinglePerimeter) -> "PerimeterSet":
        # Subtraction includes the area in the PerimeterSet
        return PerimeterSet(
            perimeters=self.perimeters + other,
            restricted_perimeters=self.restricted_perimeters,
        )

    def __sub__(self, other: SinglePerimeter) -> "PerimeterSet":
        # Subtraction excludes the area from the PerimeterSet
        return PerimeterSet(
            perimeters=self.perimeters,
            restricted_perimeters=self.restricted_perimeters + other,
        )

    def __getitem__(self, item: Label) -> SinglePerimeter:
        for perimeter in self.all_perimeters:
            if perimeter.label == item or perimeter.int_id == item:
                return perimeter
        raise KeyError(f"Item was not found, {item} amongst {self.all_perimeters}")

    @cached_property
    def video(self) -> VideoMetadata:
        result = self.all_perimeters[0].video
        for p in self.all_perimeters[1:]:
            result = VideoMetadata.join(result, p.video)
        return result

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

    @cached_property
    def centroid_meters(self):
        """
        :return: The mean of all perimeter centroids in the set
        """
        return np.mean([perimeter.centroid_meters for perimeter in self.all_perimeters], axis=0)

    def combined_framewise_confined_coordinates(self, coordinates: NDArrayFp64):
        present = np.any([perimeter.confined_coordinate_boolean_index(coordinates) for perimeter in self.perimeters])
        if self.restricted_perimeters:
            present = present & ~np.any(
                [perimeter.confined_coordinate_boolean_index(coordinates) for perimeter in self.restricted_perimeters]
            )
        return present

    def compute_confined_coordinate_boolean_index(
        self, coordinates: NDArrayFp64, manual_video: Optional[VideoMetadata] = None, ax: Axes = None, **inspect_kwargs
    ) -> np.ndarray[bool, bool]:
        result = self.combined_framewise_confined_coordinates(coordinates)

        self._post_confinement_analysis_inspect_plot(result, coordinates, manual_video, ax, **inspect_kwargs)

        return result

    def change_reference(self, **perimeter_change_reference_kwargs):
        return self.__class__(
            perimeters=tuple(
                perimeter.change_reference(**perimeter_change_reference_kwargs) for perimeter in self.perimeters
            ),
            restricted_perimeters=tuple(
                perimeter.change_reference(**perimeter_change_reference_kwargs)
                for perimeter in self.restricted_perimeters
            )
            if self.restricted_perimeters
            else None,
        )

    @property
    def perimeter_to_int_id(self):
        return {perimeter: perimeter.int_id for perimeter in self.all_perimeters}

    @property
    def int_id_to_perimeter(self):
        return {perimeter.int_id: perimeter for perimeter in self.all_perimeters}

    @property
    def label_to_perimeter(self):
        return {perimeter.label: perimeter for perimeter in self.all_perimeters}

    @property
    def int_id_to_label(self):
        return {perimeter.int_id: perimeter.label for perimeter in self.all_perimeters}

    @property
    def reference_point(self):
        expected_reference_point = self.all_perimeters[0].reference_point
        if equality := np.all(
            expected_reference_point == perimeter.reference_point for perimeter in self.all_perimeters
        ):
            logger.warning("The reference points are different within the perimeter set")
        if not equality or not np.any(expected_reference_point):
            return None
        return expected_reference_point

    @property
    def inspect_image(self):
        result = self.all_perimeters[0].inspect_image
        assert all(result == perimeter.inspect_image for perimeter in self.all_perimeters)
        return result

    @property
    def inspect_image_path(self):
        result = self.all_perimeters[0].inspect_image_path
        assert all(result == perimeter.inspect_image_path for perimeter in self.all_perimeters)
        return result

    @property
    def all_perimeters(self) -> tuple[SinglePerimeter, ...]:
        if not self.restricted_perimeters:
            return tuple(self.perimeters)
        return *self.perimeters, *self.restricted_perimeters

    @property
    def labels(self) -> tuple:
        return tuple(perimeter.label for perimeter in self.all_perimeters)

    @cached_property
    def number_of_perimeters(self) -> int:
        return len(self.all_perimeters)

    @cached_property
    def number_of_vertices(self) -> int:
        vertex_numbers = []
        for p in self.all_perimeters:
            if hasattr(p, "polygon_order"):
                vertex_numbers.append(p.polygon_order)
            elif hasattr(p, "center_pixels"):  # circle
                vertex_numbers.append(1)
        return sum(vertex_numbers)

    @property
    def get_only_perimeter(self) -> SinglePerimeter:
        assert self.number_of_perimeters == 1
        return self.all_perimeters[0]

    def plot_perimeter_on_ax(
        self, ax: Axes, inspect_pixels: bool = False, manual_resize_multiplier: Optional[float] = None, **plot_kwargs
    ) -> Axes:
        for perimeter in self.all_perimeters:
            perimeter.plot_perimeter_on_ax(ax, inspect_pixels, manual_resize_multiplier, **plot_kwargs)

        return ax

    def plot(
        self,
        manual_ax: Axes = None,
        coordinates: Optional[NDArrayFp64] = None,
        inspect_pixels: bool = False,
        **perimeter_plot_kwargs,
    ):
        if manual_ax is None:
            fig, ax = plt.subplots(constrained_layout=True)
        else:
            ax = manual_ax

        for perimeter in self.all_perimeters:
            perimeter.plot_perimeter(manual_ax=ax, inspect_pixels=inspect_pixels, **perimeter_plot_kwargs)

        if coordinates is not None:
            ax = plot_coordinates(coordinates, ax, inspect_pixels, self.video)

        if not manual_ax:
            ax.legend(**BOTTOM_LEGEND_KWARGS)
            generic_inspection_finalization(self.class_inspect_arg or True, f"0-{self.label}{INSPECT_FIG_FILE_FORMAT}")

        return ax


@validate_arguments
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
        filtered_perimeters, restricted_perimeters = [], []
        for label, perimeter in perimeters.items():
            if isinstance(label, str) and label.lower().startswith("restricted"):
                restricted_perimeters.append(perimeter)
            else:
                filtered_perimeters.append(perimeter)
        result[image_name] = PerimeterSet(perimeters=filtered_perimeters, restricted_perimeters=restricted_perimeters)
    return result
