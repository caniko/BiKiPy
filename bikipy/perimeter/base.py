from abc import abstractmethod
from functools import cached_property
from logging import getLogger
from typing import Any, ClassVar, Literal, Optional, Sequence, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from pydantic import DirectoryPath, Field, FilePath, root_validator, validate_arguments

from bikipy.core.base_class import BaseBikipyHashable
from bikipy.core.typing import NDArrayFp64, NDArrayInt16
from bikipy.core.video import VideoMetadataMixin, convert_meters_to_pixels
from bikipy.perimeter.utils import get_coco_array_from_path_or_array
from bikipy.utils.io.makesense import get_point_from_makesense_row, read_makesense_point

logger = getLogger(__name__)

StringPerimeterShapes = Literal["circle", "polygon", "rectangle"]


class BasePerimeter(BaseBikipyHashable, VideoMetadataMixin):
    impenetrable: bool = Field(
        False,
        description="Signifies the impenetrability of the perimeter. "
        "Usually because the perimeter is insurmountable or slippery",
    )
    int_id: Optional[int] = Field(description="For multi-perimeter trials where sequential confinement is used")
    group_label: Optional[str]

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
    def change_reference(self, new_reference: Optional[NDArrayFp64]):
        ...

    @abstractmethod
    def expand(self, perimeter_border_normal_meters: float | NDArrayFp64):
        ...

    @abstractmethod
    def closest_point_on_edge_to_coordinates(self, coordinates: NDArrayFp64) -> NDArrayFp64:
        ...

    @abstractmethod
    def vector_to_closest_point_on_edge(self, coordinates: NDArrayFp64) -> NDArrayFp64:
        ...

    @abstractmethod
    def plot_perimeter(
        self,
        inspect_pixels: bool = False,
        perimeter_border_normal_pixels: Optional[float] = None,
        ax: Any = None,
        include_geometric_legend: bool = False,
        colormap: Any = None,
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
        self.reference_point_array = np.asarray(value)

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
            if inspect_pixels:
                coordinates = convert_meters_to_pixels(coordinates, self.video)

            histogram, _x_edges, _y_edges = np.histogram2d(
                *coordinates[np.logical_and(*np.isfinite(coordinates).T)].T, bins=60
            )
            ax.imshow(histogram.T, interpolation="sinc")
            ax.plot(*coordinates.T, ".r-")

        ax.set_title(self.label)

        return self.plot_perimeter(
            **perimeter_plot_kwargs if perimeter_plot_kwargs else {}, ax=ax, inspect_pixels=inspect_pixels
        )


AnyPerimeter = TypeVar("AnyPerimeter", bound=BasePerimeter)


class PerimeterSet(BaseBikipyHashable):
    perimeters: list[AnyPerimeter]
    restricted_perimeters: Optional[list[AnyPerimeter]]

    label: Optional[str]

    category: ClassVar[Optional[str]] = "perimeter"

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

    def __getitem__(self, item: str | int):
        for perimeter in self.all_perimeters:
            if perimeter.label == item or perimeter.int_id == item:
                return perimeter
        raise KeyError(f"Item was not found, {item} amongst {self.all_perimeters}")

    @cached_property
    def group(self):
        grouped = {}
        for perimeter in self.all_perimeters:
            if (label := perimeter.group_label) not in grouped:
                grouped[label] = [perimeter]
            else:
                grouped[label].append(perimeter)

        # Groups with one perimeter member should be the value of the respective key
        for label, perimeters in grouped.items():
            if len(perimeters) == 1:
                grouped[label] = perimeters[0]
            else:
                grouped[label] = tuple(perimeters)

        return grouped

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

    def change_reference_with_coco(
        self,
        metadata_path: Optional[FilePath],
        coco_array: Optional[NDArrayFp64],
    ):
        coco_array = get_coco_array_from_path_or_array(metadata_path, coco_array)

        if len(coco_array) != 1:
            msg = (
                "The coco array includes more than one annotation. "
                "Please use change_reference_with_coco_with_plural_references()"
            )
            raise ValueError(msg)

        return self.__class__(
            perimeters=self.perimeters,
            restricted_perimeters=self.restricted_perimeters,
            reference_point_array=coco_array,
        )

    def change_reference_with_coco_with_plural_references(
        self,
        metadata_path: Optional[FilePath],
        coco_array: Optional[NDArrayFp64],
        map_to_image_names: bool = True,
        **kwargs,
    ):
        coco_array = get_coco_array_from_path_or_array(metadata_path, coco_array)

        perimeter_set_kwargs = {}
        for perimeter in self.perimeters:
            image_name_to_referenced_perimeters = perimeter.change_reference_with_coco_with_plural_references(
                coco_array=coco_array, **kwargs
            )
            for (
                image_name,
                referenced_perimeter,
            ) in image_name_to_referenced_perimeters.items():
                if image_name in perimeter_set_kwargs:
                    perimeter_set_kwargs[image_name]["perimeters"].append(referenced_perimeter)
                else:
                    perimeter_set_kwargs[image_name] = {"perimeters": [referenced_perimeter]}

        for perimeter in self.restricted_perimeters or []:
            image_name_to_referenced_perimeters = perimeter.change_reference_with_coco_with_plural_references(
                coco_array, **kwargs
            )
            for (
                image_name,
                referenced_perimeter,
            ) in image_name_to_referenced_perimeters.items():
                if "restricted_perimeters" in perimeter_set_kwargs[image_name]:
                    perimeter_set_kwargs[image_name]["restricted_perimeters"].append(referenced_perimeter)
                else:
                    perimeter_set_kwargs[image_name] = {"restricted_perimeters": [referenced_perimeter]}

        if map_to_image_names:
            return {
                image_name: self.__class__(**perimeter_data)
                for image_name, perimeter_data in perimeter_set_kwargs.items()
            }
        return [self.__class__(**perimeter_data) for perimeter_data in perimeter_set_kwargs.values()]

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
    def all_perimeters(self) -> tuple[AnyPerimeter, ...]:
        if not self.restricted_perimeters:
            return tuple(self.perimeters)
        return *self.perimeters, *self.restricted_perimeters

    @property
    def labels(self) -> tuple:
        return tuple(perimeter.label for perimeter in self.all_perimeters)


@validate_arguments
def perimeter_set_from_makesense(
    perimeter_path: FilePath, shape: Optional[StringPerimeterShapes], **perimeter_kwargs
) -> dict[str, PerimeterSet]:
    match shape:
        case "circle":
            from bikipy.perimeter.radial.circle import CirclePerimeter

            return CirclePerimeter.from_makesense_line(perimeter_path, **perimeter_kwargs)
        case "rectangle":
            from bikipy.perimeter.polygon.rectangle import RectanglePerimeter

            return RectanglePerimeter.from_makesense_csv_rectangle(perimeter_path, **perimeter_kwargs)
        case "triangle":
            from bikipy.perimeter.polygon.rectangle import RectanglePerimeter

            return RectanglePerimeter.from_makesense_csv_rectangle(perimeter_path, **perimeter_kwargs)
        case "polygon":
            from bikipy.perimeter.polygon.base import PolygonPerimeter

            return PolygonPerimeter.from_makesense_coco_polygon(perimeter_path, **perimeter_kwargs)
        case _:
            raise ValueError


def perimeter_set_from_image_name_to_perimeters(image_name_to_perimeters: dict[str, "AnyPerimeter"]):
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
