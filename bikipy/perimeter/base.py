from abc import abstractmethod
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, Sequence, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from pydantic import DirectoryPath, Field, FilePath, root_validator, validate_arguments

from bikipy.core.base_class import BikipyBase, BikipyBaseHashable
from bikipy.core.typing import NDArrayFp64, NDArrayInt16
from bikipy.core.video import VideoMetadata, VideoMetadataMixin
from bikipy.perimeter.utils import get_coco_array_from_path_or_array
from bikipy.utils.image import read_image
from bikipy.utils.io.makesense import read_makesense_point, get_point_from_makesense_row

logger = getLogger(__name__)

StringPerimeterShapes = Literal["circle", "parallelogram", "polygon", "rectangle"]


class BasePerimeter(BikipyBaseHashable, VideoMetadataMixin):
    impenetrable: bool = Field(
        False,
        description="Signifies the impenetrability of the perimeter. "
        "Usually because the perimeter is insurmountable or slippery",
    )

    reference_point_coco_path: Optional[FilePath] = None
    reference_point_array: Optional[NDArrayInt16] = None
    inspect_image_path: Optional[FilePath] = None
    inspect_image_array: Optional[NDArrayFp64] = None

    category: ClassVar[Optional[str]] = "perimeter"

    @abstractmethod
    def coordinate_confinement_boolean_index(self, coordinates: NDArrayFp64):
        ...

    @abstractmethod
    def change_reference(self, new_reference: Optional[NDArrayFp64], **new_inspect_image_kwargs):
        """"""
        ...

    @abstractmethod
    def plot_perimeter(
        self,
        perimeter_border_normal_pixel_magnitude: Optional[float] = None,
        ax: Any = None,
        include_geometric_legend: bool = False,
        colormap: Any = None,
    ):
        ...

    @staticmethod
    def _new_inspect_image(
        perimeter,
        new_inspect_image: Optional[NDArrayFp64] = None,
        new_inspect_image_path: Optional[FilePath] = None,
    ):
        if new_inspect_image_path:
            if not (new_inspect_image_path := Path(new_inspect_image_path)).exists():
                msg = f"new_inspect_image_path, {new_inspect_image_path}, does not exist"
                raise AttributeError(msg)
            perimeter.inspect_image = new_inspect_image_path
        elif np.any(new_inspect_image):
            perimeter.inspect_image = new_inspect_image
        else:
            perimeter.inspect_image = None
        return perimeter

    @root_validator(pre=True)
    def mutually_exclusive(cls, values):
        if all(key in values and values[key] for key in ("inspect_image_path", "inspect_image_array")):
            msg = "inspect_image_path and inspect_image_array must be defined " "mutually exclusive"
            raise AttributeError(msg)
        if all(key in values and values[key] for key in ("reference_point_coco_path", "reference_point_array")):
            msg = "reference_point_coco_path and reference_point_array must be " "defined mutually exclusive"
            raise AttributeError(msg)
        return values

    @property
    def inspect_image(self):
        if self.inspect_image_array is None and not self.inspect_image_path:
            return None
        return read_image(self.inspect_image_path) if self.inspect_image_path else self.inspect_image_array

    @inspect_image.setter
    def inspect_image(self, value):
        self.inspect_image_array = np.asarray(value)

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
        metadata_path: Optional[FilePath] = None,
        coco_array: Optional[NDArrayFp64] = None,
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
        metadata_path: Optional[FilePath] = None,
        coco_array: Optional[NDArrayFp64] = None,
        image_root: Optional[DirectoryPath] = None,
        map_to_image_names: bool = True,
    ):
        def _change_reference_loop_func(reference_point, img_name):
            return self.change_reference(
                reference_point,
                new_inspect_image_path=image_root / img_name if image_root else None,
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

    @classmethod
    def detect_sequential_border_presence(
        cls,
        coordinates: NDArrayFp64,
        superior_poly_border_instances: Optional[Sequence],
        inferior_poly_border_instances: Optional[Sequence] = None,
        clean_outliers: bool = True,
    ):
        """
        Define sequential perimeter confinements of coordinates

        Parameters
        ----------
        coordinates: NDArrayFp64
            Coordinates that will have their confinement tested

        superior_poly_border_instances: Sequence
            PolygonPerimeter instances that will have the highest priority
            in case of overlap with respect to confinement

        inferior_poly_border_instances: Sequence
            PolygonPerimeter instances that will have the lowest priority
            in case of overlap with respect to confinement

        clean_outliers
            Clear elements that aren't confined to any of the given border_corners
            as a final action before returning the sequential perimeter presence

        Returns
        -------
        NDArrayFp64 that stores the sequential perimeter presence across frames
        """

        coordinates = np.asarray(coordinates)

        perimeter_sequence = (
            (*inferior_poly_border_instances, *superior_poly_border_instances)
            if inferior_poly_border_instances
            else superior_poly_border_instances
        )
        presence = np.zeros(
            coordinates.shape[0],
            dtype=np.uint8 if len(perimeter_sequence) <= 255 else np.uint16,
        )

        overlap_locations = {}
        for perimeter in perimeter_sequence:
            confined_coord_booleans_index = perimeter.coordinate_confinement_boolean_index(coordinates)

            if presence[confined_coord_booleans_index].any():
                overlap_locations[perimeter.label] = np.flatnonzero(presence[confined_coord_booleans_index])
                presence[overlap_locations[perimeter.label]] = 0
                logger.info(
                    f"BasePerimeter {perimeter.label} has coordinate overlap with "
                    f"other border_corners, {overlap_locations[perimeter.label].size}"
                )

            presence[confined_coord_booleans_index] = perimeter.int_id

        valid_indices = np.nonzero(presence)
        if clean_outliers:
            presence = presence[valid_indices]

        boolean_array = np.full(coordinates.shape[0], False, dtype=np.bool)
        boolean_array[valid_indices] = True

        return presence, valid_indices, boolean_array

    def apply_label_prefix_suffix(self, prefix: Optional[str] = None, suffix: Optional[str] = None) -> None:
        if prefix:
            self.label = f"{prefix}_{self.label}"
        if suffix:
            self.label = f"{self.label}_{suffix}"

    def plot(
        self,
        ax: Any = None,
        coordinates: Optional[NDArrayFp64] = None,
        perimeter_plot_kwargs: Optional[dict] = None,
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

        if self.inspect_image is not None:
            ax.imshow(self.inspect_image)

        if coordinates is not None:
            coordinates = np.asarray(coordinates)

            histogram, _x_edges, _y_edges = np.histogram2d(
                *coordinates[np.logical_and(*np.isfinite(coordinates).T)].T, bins=60
            )
            ax.imshow(histogram.T, interpolation="sinc")
            ax.plot(*coordinates.T, ".r-")

        ax.set_title(self.best_id)

        return self.plot_perimeter(**perimeter_plot_kwargs if perimeter_plot_kwargs else {}, ax=ax)

    @staticmethod
    def perimeter_set_from_image_name_to_perimeters(image_name_to_perimeters: "dict[str, AnyPerimeter]"):
        result = {}
        for image_name, perimeters in image_name_to_perimeters.items():
            filtered_perimeters, restricted_perimeters = [], []
            for label, perimeter in perimeters.items():
                if isinstance(label, str) and label.lower().startswith("restricted"):
                    restricted_perimeters.append(perimeter)
                else:
                    filtered_perimeters.append(perimeter)
            result[image_name] = PerimeterSet(
                perimeters=filtered_perimeters, restricted_perimeters=restricted_perimeters
            )
        return result


AnyPerimeter = TypeVar("AnyPerimeter", bound=BasePerimeter)


class PerimeterSet(BikipyBase):
    perimeters: list[AnyPerimeter]
    restricted_perimeters: Optional[list[AnyPerimeter]] = None

    category: ClassVar[Optional[str]] = "perimeter"

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
            result[perimeter.best_id or i] = framewise_confined_coordinates
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
        metadata_path: Optional[FilePath] = None,
        coco_array: Optional[NDArrayFp64] = None,
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
        metadata_path: Optional[FilePath] = None,
        coco_array: Optional[NDArrayFp64] = None,
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

    def apply_label_prefix_suffix(self, prefix: Optional[str] = None, suffix: Optional[str] = None) -> None:
        for perimeter in self.all_perimeters:
            perimeter.apply_label_prefix_suffix(prefix, suffix)

    @cached_property
    def perimeter_to_int_id(self):
        return {perimeter: perimeter.int_id for perimeter in self.all_perimeters}

    @cached_property
    def perimeter_to_label(self):
        return {perimeter: perimeter.label for perimeter in self.all_perimeters}

    @cached_property
    def int_id_to_perimeter(self):
        return {perimeter.int_id: perimeter for perimeter in self.all_perimeters}

    @cached_property
    def label_to_perimeter(self):
        return {perimeter.label: perimeter for perimeter in self.all_perimeters}

    @cached_property
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
            return self.perimeters
        return *self.perimeters, *self.restricted_perimeters

    @property
    def labels(self) -> tuple:
        return tuple(perimeter.label for perimeter in self.all_perimeters)


@validate_arguments
def perimeter_set_from_makesense(
    perimeter_path: FilePath, shape: Optional[StringPerimeterShapes] = None, **perimeter_kwargs
) -> dict[str, PerimeterSet]:
    match shape:
        case "circle":
            from bikipy.perimeter.radial.circle import CirclePerimeter

            return CirclePerimeter.from_makesense_line(perimeter_path, **perimeter_kwargs)
        case "rectangle" | "parallelogram":
            from bikipy.perimeter.polygon.parallelogram import ParallelogramPerimeter

            return ParallelogramPerimeter.from_makesense_csv_rectangle(perimeter_path, **perimeter_kwargs)
        case "polygon":
            from bikipy.perimeter.polygon.base import PolygonPerimeter

            return PolygonPerimeter.from_makesense_coco_polygon(perimeter_path, **perimeter_kwargs)
        case _:
            raise ValueError
