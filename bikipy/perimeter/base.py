from functools import cached_property
from logging import getLogger
from typing import Any, ClassVar, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from pydantic import FilePath, root_validator

from bikipy.core.base_class import BikipyBase, BikipyBaseHashable
from bikipy.perimeter.utils import get_coco_array_from_path_or_array
from bikipy.utils.misc import read_image
from bikipy.utils.typing import NDArray

logger = getLogger(__name__)


class BasePerimeter(BikipyBaseHashable):
    reference_point_coco_path: Optional[FilePath] = None
    reference_point_array: Optional[NDArray] = None
    inspect_image_path: Optional[FilePath] = None
    inspect_image_array: Optional[NDArray] = None

    category: ClassVar[Optional[str]] = "perimeter"

    @root_validator(pre=True)
    def mutually_exclusive(cls, values):
        if all(key in values for key in ("inspect_image_path", "inspect_image_array")):
            msg = (
                "inspect_image_path and inspect_image_array must be defined "
                "mutually exclusive"
            )
            raise AttributeError(msg)
        if all(
            key in values
            for key in ("reference_point_coco_path", "reference_point_array")
        ):
            msg = (
                "reference_point_coco_path and reference_point_array must be "
                "defined mutually exclusive"
            )
            raise AttributeError(msg)
        return values

    @property
    def inspect_image(self):
        if self.inspect_image_array is None and not self.inspect_image_path:
            return None
        return (
            read_image(self.inspect_image_path)
            if self.inspect_image_path
            else self.inspect_image_array
        )

    @inspect_image.setter
    def inspect_image(self, value):
        self.inspect_image_array = np.asarray(value)

    @property
    def reference_point(self):
        from bikipy.perimeter.io.makesense import reference_point_from_coco_path

        if self.reference_point_array is None and not self.reference_point_coco_path:
            return None
        return (
            reference_point_from_coco_path(self.reference_point_coco_path)
            if self.reference_point_array is None
            else self.reference_point_array
        )

    @reference_point.setter
    def reference_point(self, value):
        self.reference_point_array = np.asarray(value)

    def plot(self, ax: Any = None, coordinates: Optional[Sequence] = None):
        """
        Plot the perimeter using matplotlib. Optionally, plot coordinates alongside the perimeter

        Parameters
        ----------
        ax
            Axes object that the plot will be saved in. A new instance of Axes will be used
            if object returns False.
        coordinates
            Sequence of 2D coordinates that will be plotted alongside the perimeter

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

        return ax


class PerimeterSet(BikipyBase):
    perimeters: tuple
    restricted_perimeters: Optional[tuple] = None

    category: ClassVar[Optional[str]] = "perimeter"

    def __getitem__(self, item: Union[str, int]):
        for perimeter in self._all_perimeters:
            if perimeter.label == item or perimeter.int_id == item:
                return perimeter
        raise KeyError(f"Item was not found, {item} amongst {self._all_perimeters}")

    @cached_property
    def group(self):
        grouped = {}
        for perimeter in self._all_perimeters:
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
        return np.mean([perimeter.centroid for perimeter in self.perimeters], axis=0)

    def discrete_confined_coordinates(
        self, coordinates: Sequence, inspect: bool = False
    ):
        ax = self.plot() if inspect else None
        result = {}
        for i, perimeter in enumerate(self._all_perimeters):
            confined_coordinates = perimeter.confined_coordinates(coordinates, ax=ax)
            result[perimeter.best_id or i] = confined_coordinates
        if inspect:
            plt.show()
        return result

    def combined_confined_coordinates(self, coordinates: Sequence):
        present = np.any(
            [
                perimeter.coordinate_confinement_boolean_index(coordinates)
                for perimeter in self.perimeters
            ]
        )
        if self.restricted_perimeters:
            present = present & ~np.any(
                [
                    perimeter.coordinate_confinement_boolean_index(coordinates)
                    for perimeter in self.restricted_perimeters
                ]
            )
        return present

    def change_reference(self, **perimeter_change_reference_kwargs):
        return self.__class__(
            perimeters=tuple(
                perimeter.change_reference(**perimeter_change_reference_kwargs)
                for perimeter in self.perimeters
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
        coco_array: Optional[np.ndarray] = None,
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
        coco_array: Optional[np.ndarray] = None,
        map_to_image_names: bool = True,
        **kwargs,
    ):
        coco_array = get_coco_array_from_path_or_array(metadata_path, coco_array)

        perimeter_set_kwargs = {}
        for perimeter in self.perimeters:
            image_name_vs_referenced_perimeters = (
                perimeter.change_reference_with_coco_with_plural_references(
                    coco_array=coco_array, **kwargs
                )
            )
            for (
                image_name,
                referenced_perimeter,
            ) in image_name_vs_referenced_perimeters.items():
                if image_name in perimeter_set_kwargs:
                    perimeter_set_kwargs[image_name]["perimeters"].append(
                        referenced_perimeter
                    )
                else:
                    perimeter_set_kwargs[image_name] = {
                        "perimeters": [referenced_perimeter]
                    }

        for perimeter in self.restricted_perimeters or []:
            image_name_vs_referenced_perimeters = (
                perimeter.change_reference_with_coco_with_plural_references(
                    coco_array, **kwargs
                )
            )
            for (
                image_name,
                referenced_perimeter,
            ) in image_name_vs_referenced_perimeters.items():
                if "restricted_perimeters" in perimeter_set_kwargs[image_name]:
                    perimeter_set_kwargs[image_name]["restricted_perimeters"].append(
                        referenced_perimeter
                    )
                else:
                    perimeter_set_kwargs[image_name] = {
                        "restricted_perimeters": [referenced_perimeter]
                    }

        if map_to_image_names:
            return {
                image_name: self.__class__(**perimeter_data)
                for image_name, perimeter_data in perimeter_set_kwargs.items()
            }
        return [
            self.__class__(**perimeter_data)
            for perimeter_data in perimeter_set_kwargs.values()
        ]

    # def plot(self, **kwargs):
    #     ax = super().plot(**kwargs)
    #     return PolygonPerimeter.plot_perimeters(self.perimeters, ax)

    @cached_property
    def perimeter_vs_int_id(self):
        return {perimeter: perimeter.int_id for perimeter in self.perimeters}

    @cached_property
    def perimeter_vs_labels(self):
        return {perimeter: perimeter.labels for perimeter in self.perimeters}

    @cached_property
    def int_id_vs_label(self):
        return {perimeter.int_id: perimeter.labels for perimeter in self.perimeters}

    @property
    def reference_point(self):
        expected_reference_point = self._all_perimeters[0].reference_point
        if equality := np.all(
            expected_reference_point == perimeter.reference_point
            for perimeter in self._all_perimeters
        ):
            logger.warning(
                "The reference points are different within the perimeter set"
            )
        if not equality or not np.any(expected_reference_point):
            return None
        return expected_reference_point

    @property
    def inspect_image(self):
        result = self.perimeters[0].inspect_image
        assert all(
            result == perimeter.inspect_image for perimeter in self._all_perimeters
        )
        return result

    @property
    def inspect_image_path(self):
        result = self.perimeters[0].inspect_image_path
        assert all(
            result == perimeter.inspect_image_path for perimeter in self._all_perimeters
        )
        return result

    @property
    def _all_perimeters(self) -> Sequence:
        if not self.restricted_perimeters:
            return self.perimeters
        return (
            *self.perimeters,
            *self.restricted_perimeters,
        )
