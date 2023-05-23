from functools import cached_property, lru_cache
from logging import getLogger
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic_numpy import NDArray
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64

from bikipy._constant import INSPECT_FIG_FILE_FORMAT
from bikipy.behaviour.core.base import HabituationTrialMixin
from bikipy.behaviour.core.enclosure.base import EnclosedExperiment, EnclosedTrial
from bikipy.behaviour.core.enclosure.quadrant import Quadrant
from bikipy.behaviour.utils import (
    blanket_enclosed_experiment_label_generator,
    reduce_repeating_sequences,
)
from bikipy.feature.motion import get_combined_features_from_merged_motion_island_data
from bikipy.perimeter import RectanglePerimeter
from bikipy.utils.collection_utils import generic_multi_indexer
from bikipy.utils.math.inside.polygon import parallel_point_inside_polygon
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS
from bikipy.utils.plot.inspect import generic_inspection_finalization

logger = getLogger(__name__)
quadrant_grid_typing = tuple[int, int]

A = 1


def motion_analysis_indexer_for_subsection(category: Any, level: int):
    return generic_multi_indexer(
        "Displacement", "MedianSpeed", "MedianAcceleration", "FreezingTime", "Entries", "SecondsPresent"
    )(category, level)


class RectangleEnclosedExperiment(EnclosedExperiment):
    pass


class RectangleEnclosedTrial(EnclosedTrial):
    rectangle_2d_bin: quadrant_grid_typing = (2, 2)

    trial_perimeter_enclosure_class = RectanglePerimeter

    manual_center_rectangle_dimensions_meters: Optional[NDArrayFp64]
    center_rectangle_dimensions_to_spatial_resolution_ratio: Optional[float]

    @cached_property
    def _center_periphery_is_defined(self) -> bool:
        return (
            self.manual_center_rectangle_dimensions_meters is not None
            or self.center_rectangle_dimensions_to_spatial_resolution_ratio
        )

    @cached_property
    def _inspect_center_periphery_directory(self):
        result = self.inspect_arg / "center_periphery"
        result.mkdir(exist_ok=True)
        return result

    @cached_property
    def _inspect_quadrant_directory(self):
        result = self.inspect_arg / "quadrant"
        result.mkdir(exist_ok=True)
        return result

    @cached_property
    def quadrant_grid_coordinate_to_vertices(self) -> dict[quadrant_grid_typing, NDArrayFp64]:
        horizontal_uniform_distance = self.video.metric_horizontal_resolution / self.rectangle_2d_bin[0]
        vertical_uniform_distance = self.video.metric_vertical_resolution / self.rectangle_2d_bin[1]
        result = {}
        for h in range(1, self.rectangle_2d_bin[0] + 1):
            horizontal_coordinate_min = horizontal_uniform_distance * (h - 1)
            horizontal_coordinate_max = horizontal_uniform_distance * h
            for v in range(1, self.rectangle_2d_bin[1] + 1):
                vertical_coordinate_min = vertical_uniform_distance * (v - 1)
                vertical_coordinate_max = vertical_uniform_distance * v

                quadrant = np.array(
                    (
                        (horizontal_coordinate_min, vertical_coordinate_min),
                        (horizontal_coordinate_max, vertical_coordinate_min),
                        (horizontal_coordinate_max, vertical_coordinate_max),
                        (horizontal_coordinate_min, vertical_coordinate_max),
                    )
                )

                if self.center_meter_translation is not None:
                    quadrant += self.center_meter_translation

                result[(h, v)] = quadrant

        return result

    @cached_property
    def quadrant_grid_coordinates(self) -> tuple[tuple[int, int], ...]:
        return tuple(self.quadrant_grid_coordinate_to_vertices)

    @cached_property
    def quadrant_index_to_quadrant_grid_coordinate(self) -> dict[int, quadrant_grid_typing]:
        return {
            i: quadrant_grid_coordinate
            for i, quadrant_grid_coordinate in enumerate(self.quadrant_grid_coordinate_to_vertices, start=1)
        }

    @cached_property
    def quadrant_grid_coordinate_to_quadrant_index(self) -> dict[quadrant_grid_typing, int]:
        return {
            quadrant_grid_coordinate: i
            for i, quadrant_grid_coordinate in self.quadrant_index_to_quadrant_grid_coordinate.items()
        }

    @cached_property
    def quadrant_grid_coordinate_to_quadrant(self) -> dict[quadrant_grid_typing, Quadrant]:
        """
        Left to right, top to down
        :return:
        """
        result = {
            quadrant_grid_coordinate: Quadrant(
                vertices_in_meters=self.quadrant_grid_coordinate_to_vertices[quadrant_grid_coordinate],
                kinematic_coordinates=self.reader.kinematic_coordinates,
                fps=self.video.fps,
                quadrant_index=quadrant_index,
            )
            for quadrant_index, quadrant_grid_coordinate in self.quadrant_index_to_quadrant_grid_coordinate.items()
        }
        if self.inspect_arg:
            fig, ax = self.video.subplot()
            ax.set_title(f"Quadrants_Trial_#{self.label}")

            coordinates = self.video.prepare_coordinates_for_plotting(self.reader.kinematic_coordinates)

            confined = np.zeros(self.reader.frames, dtype=bool)

            colors = plt.cm.rainbow(np.linspace(0, 1, len(result) + 1))
            for color, (grid_coordinate, quadrant) in zip(colors, result.items()):
                ax.plot(
                    *quadrant.plot_vertices(self.video).T,
                    label=f"({grid_coordinate[0]}, {grid_coordinate[1]})",
                    color=color,
                )
                ax.scatter(*coordinates[quadrant.confinement_boolean_index].T, color=color)

                confined = confined | quadrant.confinement_boolean_index

            ax.scatter(*coordinates[~confined].T, color=colors[-1], label="Unconfined")
            ax.scatter(*self.video.center_for_plot.T, color="r", label="VideoCenter")

            if self.manual_center_meters is not None:
                manual_center = self.manual_center_meters
                if self.video.coordinates_need_to_be_scaled_for_plot:
                    manual_center = manual_center * self.video.pixels_per_meter

                ax.scatter(*manual_center.T, color="k", label="ManualCenter")

            plt.legend(**BOTTOM_LEGEND_KWARGS)

            generic_inspection_finalization(
                self._inspect_quadrant_directory / f"{self.label}{INSPECT_FIG_FILE_FORMAT}",
                debug_save_message=f"Saved perimeter_set {self.label} inspect plot to {self.class_inspect_arg}",
            )

        return result

    @cached_property
    def location_sequence_quadrant(self) -> NDArrayFp64:
        raw_location_sequence_quadrant = np.zeros(self.number_of_frames, dtype=np.uint8)
        for quadrant_index, quadrant_grid_coordinate in self.quadrant_index_to_quadrant_grid_coordinate.items():
            quadrant = self.quadrant_grid_coordinate_to_quadrant[quadrant_grid_coordinate]

            if np.any(raw_location_sequence_quadrant[quadrant.confinement_boolean_index]):
                logger.warning("Quadrant confinement has temporal-spatial collision with another, ignoring")

            raw_location_sequence_quadrant[quadrant.confinement_boolean_index] = quadrant_index

        return np.array(reduce_repeating_sequences(raw_location_sequence_quadrant, round(self.video.fps * 0.35)))

    @cached_property
    def quadrant_grid_coordinate_to_entries(self) -> dict[quadrant_grid_typing, int]:
        result = {}
        for quadrant_index, quadrant_grid_coordinate in self.quadrant_index_to_quadrant_grid_coordinate.items():
            result[quadrant_grid_coordinate] = np.sum(self.location_sequence_quadrant == quadrant_index)
        return result

    @cached_property
    def quadrant_grid_coordinate_to_seconds_present(self) -> dict[quadrant_grid_typing, float]:
        return {
            quadrant_grid_coordinate: quadrant.seconds_present
            for quadrant_grid_coordinate, quadrant in self.quadrant_grid_coordinate_to_quadrant.items()
        }

    # Center vs Periphery ==============================================================
    @cached_property
    def center_rectangle_dimensions_meters(self) -> NDArrayFp64 | None:
        if self._center_periphery_is_defined is None:
            return None
        if self.manual_center_rectangle_dimensions_meters is not None:
            return self.manual_center_rectangle_dimensions_meters
        if self.center_rectangle_dimensions_to_spatial_resolution_ratio is not None:
            return self.video.metric_resolution / self.center_rectangle_dimensions_to_spatial_resolution_ratio

    @cached_property
    def center_rectangle_vertices(self) -> NDArrayFp64:
        if not self._center_periphery_is_defined:
            msg = (
                "manual_center_rectangle_dimensions_meters or center_rectangle_dimensions_to_spatial_resolution_ratio "
                "must be defined for center_rectangle_vertices to be defined"
            )
            raise AttributeError(msg)

        center_point_to_center_rectangle_side_normal_lengths = self.center_rectangle_dimensions_meters / 2.0

        center = self.video.center_meters if self.manual_center_meters is None else self.manual_center_meters

        # The Y-axis is max at the image origin, hence the inversion WRT the X-axis:
        x_long, y_short = center + center_point_to_center_rectangle_side_normal_lengths
        x_short, y_long = center - center_point_to_center_rectangle_side_normal_lengths

        return np.array(((x_short, y_short), (x_short, y_long), (x_long, y_long), (x_long, y_short)))

    @cached_property
    def center_boolean_index(self) -> NDArrayBool:
        return parallel_point_inside_polygon(
            self.reader.kinematic_coordinates,
            self.center_rectangle_vertices,
            inspect_arg=self.class_inspect_arg,
            potential_label=f"{self.label}{INSPECT_FIG_FILE_FORMAT}",
            video=self.video,
        )

    @cached_property
    def periphery_boolean_index(self) -> NDArrayBool:
        return ~self.center_boolean_index

    @cached_property
    def motion_center(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.center_boolean_index,
            self.reader.kinematic_coordinates,
            self.video.fps,
        )

    @cached_property
    def motion_periphery(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.periphery_boolean_index,
            self.reader.kinematic_coordinates,
            self.video.fps,
        )

    @cached_property
    def location_sequence_center_periphery(self) -> NDArray:
        # 1 is center, 2 is periphery, 0 is unknown
        location_sequence_center_periphery = np.zeros_like(self.center_boolean_index, dtype=np.uint8)
        location_sequence_center_periphery[self.center_boolean_index] = 1
        location_sequence_center_periphery[self.periphery_boolean_index] = 2
        return np.array(
            reduce_repeating_sequences(
                location_sequence_center_periphery,
                frame_tolerance=self._frame_tolerance,
            )
        )

    @cached_property
    def center_entries(self) -> int:
        return np.sum(self.location_sequence_center_periphery == 1)

    @cached_property
    def periphery_entries(self) -> int:
        return np.sum(self.location_sequence_center_periphery == 2)

    @cached_property
    def seconds_on_center(self) -> int:
        return np.sum(self.center_boolean_index) / self.video.fps

    @cached_property
    def seconds_on_periphery(self) -> int:
        return np.sum(self.periphery_boolean_index) / self.video.fps

    @property
    def _analysis_series_list(self) -> list[pd.Series]:
        upstream_list = super()._analysis_series_list
        if self.video.resolution is None:
            return upstream_list

        data = [
            # self.gaussian_center_to_periphery_score,
        ]

        if self._center_periphery_is_defined:
            data.extend(
                (
                    *self.motion_center.values(),
                    self.center_entries,
                    self.seconds_on_center,
                    *self.motion_periphery.values(),
                    self.periphery_entries,
                    self.seconds_on_periphery,
                )
            )

        for (qgc_i, quadrant), (qgc_ii, entries) in zip(
            self.quadrant_grid_coordinate_to_quadrant.items(), self.quadrant_grid_coordinate_to_entries.items()
        ):
            assert qgc_i == qgc_ii
            data.extend((*quadrant.motion.values(), entries, quadrant.seconds_present))

        upstream_list.append(
            pd.Series(
                data,
                index=motion_column_headers(self.quadrant_grid_coordinates, 2, self._center_periphery_is_defined),
            )
        )

        return upstream_list


class RectangleEnclosedHabituationTrial(HabituationTrialMixin, RectangleEnclosedTrial):
    # TODO: Fix incorrect trial_label, when using this class
    trial_label = "Habituation"


class BlanketRectangleEnclosedTrial(RectangleEnclosedTrial):
    trial_label = "blanket_rectangle_enclosed_trial"
    excel_sheet_name = "Rectangle enclosed"

    experiment_class_name = "BlanketRectangleEnclosedExperiment"


class BlanketRectangleEnclosedExperiment(RectangleEnclosedExperiment):
    experiment_labels = blanket_enclosed_experiment_label_generator("rectangle")

    habituation_trial_class = RectangleEnclosedHabituationTrial
    trial_sequence = (BlanketRectangleEnclosedTrial,)


@lru_cache
def motion_column_headers(
    quadrant_grid_coordinates: tuple[tuple[int, int], ...], column_index_levels: int, center_periphery_is_defined: bool
) -> list[tuple]:
    result = [
        # ("Gaussian", "CenterToPeriphery")
    ]

    if center_periphery_is_defined:
        result.extend(
            (
                *motion_analysis_indexer_for_subsection("Center", column_index_levels),
                *motion_analysis_indexer_for_subsection("Periphery", column_index_levels),
            )
        )

    for quadrant_grid_coordinate in quadrant_grid_coordinates:
        category = f"Quadrant{quadrant_grid_coordinate}"
        result.extend(motion_analysis_indexer_for_subsection(category, 2))

    return result
