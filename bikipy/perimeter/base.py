from abc import abstractmethod
from collections import defaultdict
from functools import cached_property, reduce, partial
from logging import getLogger
from typing import Any, ClassVar, Literal, Optional, TypeVar

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from pydantic import DirectoryPath, Field, FilePath, root_validator, validate_arguments, PositiveInt

from bikipy.core.base_class import BaseBikipyHashable, BaseBikipyInspectMixin
from bikipy.core.typing import NDArrayFp64, NDArrayInt16
from bikipy.core.video import (
    VideoMetadataMixin,
    VideoMetadata,
)
from bikipy.perimeter.polygon.makesense import (
    init_polygon_from_makesense_coco_polygon,
    init_polygon_from_makesense_csv_rectangle,
)
from bikipy.perimeter.utils import get_coco_array_from_path_or_array
from bikipy.utils.collection_utils import evenly_spaced_indices_from_sequence
from bikipy.utils.io.makesense import get_point_from_makesense_row, read_makesense_point
from bikipy.utils.misc import plot_coordinates

logger = getLogger(__name__)

StringPerimeterShapes = Literal["circle", "polygon", "rectangle"]


class BasePerimeter(BaseBikipyHashable):
    pass


Perimeter = TypeVar("Perimeter", bound=BasePerimeter)


class BaseSinglePerimeter(BasePerimeter, BaseBikipyInspectMixin, VideoMetadataMixin):
    impenetrable: bool = Field(
        False,
        description="Signifies the impenetrability of the perimeter. "
        "Usually because the perimeter is insurmountable or slippery",
    )
    int_id: Optional[int] = Field(description="For multi-perimeter trials where sequential confinement is used")
    group_label: Optional[str]

    makesense_image_name: Optional[str]

    reference_point_coco_path: Optional[FilePath]
    reference_point_array: Optional[NDArrayInt16]

    category: ClassVar[Optional[str]] = "perimeter"
    required_video_metadata_fields = {"recording_resolution"}

    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.append(self.int_id)
        return result

    @abstractmethod
    def coordinate_confinement_boolean_index(self, coordinates: NDArrayFp64):
        ...

    @abstractmethod
    def change_reference(self, new_reference: NDArrayFp64, makesense_image_name: Optional[str] = None):
        ...

    @abstractmethod
    def expand(self, perimeter_border_normal_meters: float | NDArrayFp64):
        ...

    @abstractmethod
    def closest_point_on_edge_to_coordinates(self, coordinates: NDArrayFp64) -> NDArrayFp64:
        ...

    def inspect_closest_point_on_edge_to_coordinates(self, result: NDArrayFp64, coordinates: NDArrayFp64):
        if self.inspect:
            sb.set_theme(style="darkgrid")
            fig, ax = plt.subplots(dpi=500)

            self.plot_perimeter(ax=ax)

            with sb.color_palette("Spectral", n_colors=5):
                for i in evenly_spaced_indices_from_sequence(coordinates, 5):
                    ax.plot(*np.vstack((result[i], coordinates[i])).T)

            plt.savefig(self.inspect_directory / f"{self.label}.jpeg")

    @abstractmethod
    def vector_to_closest_point_on_edge(self, coordinates: NDArrayFp64) -> NDArrayFp64:
        ...

    @abstractmethod
    def gaze_direction_filter(self, *args, **kwargs):
        ...

    @abstractmethod
    def plot_perimeter(
        self,
        inspect_pixels: bool = False,
        perimeter_border_normal_pixels: Optional[float] = None,
        manual_ax: Any = None,
        **plot_kwargs,
    ):
        ...

    @root_validator(pre=True)
    def mutually_exclusive(cls, values):
        if all(key in values and values[key] for key in ("reference_point_coco_path", "reference_point_array")):
            msg = "reference_point_coco_path and reference_point_array must be " "defined mutually exclusive"
            raise AttributeError(msg)
        return values

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

    @validate_arguments
    def framewise_confined_coordinates(self, coordinates: NDArrayFp64, inspect: bool = False, ax: Any = None):
        """
        self.framewise_confined_coordinates to fetch confined coordinates within
        the respective perimeter

        Parameters
        ----------
        coordinates
            Coordinates that will have their confinement tested
        inspect
            If True, plot the confined coordinates
        ax

        Returns
        -------

        """
        coordinate_confinement_boolean_index = coordinates[self.coordinate_confinement_boolean_index(coordinates)]
        if inspect or ax:
            if not ax:
                ax = self.plot_self()
            ax.scatter(
                coordinate_confinement_boolean_index.T[0],
                coordinate_confinement_boolean_index.T[1],
                marker="x",
            )
            ax.set_tittle("Confined coordinates")
            plt.show()

        return coordinate_confinement_boolean_index

    def plot(
        self,
        ax: Any = None,
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
            fig, ax = plt.subplots()
            ax.set_title(self.label)

        if self.video.frame is not None:
            ax.imshow(self.video.frame)

        if coordinates is not None:
            ax = plot_coordinates(coordinates, ax, inspect_pixels, self.video)

        ax.set_title(self.label)

        return self.plot_perimeter(
            **perimeter_plot_kwargs if perimeter_plot_kwargs else {}, ax=ax, inspect_pixels=inspect_pixels
        )


SinglePerimeter = TypeVar("SinglePerimeter", bound=BaseSinglePerimeter)


class PerimeterSet(BasePerimeter, BaseBikipyInspectMixin):
    perimeters: list[SinglePerimeter]
    restricted_perimeters: Optional[list[SinglePerimeter]]

    label: Optional[str]

    category: ClassVar[Optional[str]] = "perimeter"
    _class_inspect_directory_name = "perimeter_set"

    @property
    def _to_hash(self) -> list:
        result = super()._to_hash
        result.extend(perimeter._to_hash for perimeter in self.all_perimeters)
        return result

    def __add__(self, other):
        return PerimeterSet(
            perimeters=self.perimeters + other.perimeters,
            restricted_perimeters=self.restricted_perimeters + other.restricted_perimeters,
        )

    def __getitem__(self, item: str | PositiveInt):
        for perimeter in self.all_perimeters:
            if perimeter.label == item or perimeter.int_id == item:
                return perimeter
        raise KeyError(f"Item was not found, {item} amongst {self.all_perimeters}")

    @cached_property
    def video(self) -> VideoMetadata:
        return self.all_perimeters[0].video

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
    def centroid(self):
        """
        :return: The mean of all perimeter centroids in the set
        """
        return np.mean([perimeter.centroid for perimeter in self.all_perimeters], axis=0)

    def discrete_framewise_confined_coordinates(self, coordinates: NDArrayFp64, inspect: bool = False):
        ax = self.plot() if inspect else None
        result = {}
        for i, perimeter in enumerate(self.all_perimeters):
            framewise_confined_coordinates = perimeter.framewise_confined_coordinates(coordinates, ax=ax)
            result[perimeter.label or i] = framewise_confined_coordinates
        if inspect:
            plt.show()
        return result

    def combined_framewise_confined_coordinates(self, coordinates: NDArrayFp64):
        present = np.any([perimeter.coordinate_confinement_boolean_index(coordinates) for perimeter in self.perimeters])
        if self.restricted_perimeters:
            present = present & ~np.any(
                [
                    perimeter.coordinate_confinement_boolean_index(coordinates)
                    for perimeter in self.restricted_perimeters
                ]
            )
        return present

    def coordinate_confinement_boolean_index(self, coordinates: NDArrayFp64):
        return self.combined_framewise_confined_coordinates(coordinates)

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

    @property
    def get_only_perimeter(self) -> SinglePerimeter:
        assert self.number_of_perimeters == 1
        return self.all_perimeters[0]

    def plot(
        self,
        manual_ax: Any = None,
        coordinates: Optional[NDArrayFp64] = None,
        inspect_pixels: bool = False,
        **perimeter_plot_kwargs,
    ):
        if manual_ax is None:
            fig, ax = plt.subplots(constrained_layout=True)
        else:
            ax = manual_ax

        for perimeter in self.all_perimeters:
            perimeter.plot_perimeter(manual_ax=ax, **perimeter_plot_kwargs)

        if coordinates is not None:
            ax = plot_coordinates(coordinates, ax, inspect_pixels, self.video)

        if manual_ax:
            pass
        elif self.inspect_directory:
            plt.savefig(self.class_inspect_directory / f"{self.label}.jpg")
            logger.debug(f"Saved perimeter_set {self.label} inspect plot to {self.class_inspect_directory}")
        else:
            plt.legend()
            plt.show()

        return ax


@validate_arguments
def perimeter_set_from_makesense(
    perimeter_path: FilePath, shape: Optional[StringPerimeterShapes], **perimeter_kwargs
) -> dict[str, PerimeterSet]:
    match shape:
        case "circle":
            from bikipy.perimeter.radial.circle import CirclePerimeter

            return CirclePerimeter.from_makesense_line(perimeter_path, **perimeter_kwargs)
        case "rectangle":
            return init_polygon_from_makesense_csv_rectangle(perimeter_path, **perimeter_kwargs)
        case "triangle":
            return init_polygon_from_makesense_coco_polygon(perimeter_path, **perimeter_kwargs)
        case "polygon":
            return init_polygon_from_makesense_coco_polygon(perimeter_path, **perimeter_kwargs)
        case _:
            raise ValueError


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
