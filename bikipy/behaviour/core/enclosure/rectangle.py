from functools import cached_property, lru_cache
from logging import getLogger
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import DirectoryPath, computed_field
from pydantic_numpy.typing import (
    Np1DArrayBool,
    Np1DArrayFp64,
    Np1DArrayUint8,
    Np2DArrayFp64,
)

from bikipy._constant import INSPECT_FIG_FILE_FORMAT
from bikipy.behaviour.core.base import HabituationTrialMixin
from bikipy.behaviour.core.constant import ExperimentStage
from bikipy.behaviour.core.enclosure.base import EnclosedExperiment, EnclosedTrial
from bikipy.behaviour.core.enclosure.quadrant import Quadrant
from bikipy.behaviour.utils import blanket_enclosed_experiment_label_generator
from bikipy.feature.motion import TruthIslandMetadata, merge_motion_island_data
from bikipy.math.discrete import (
    boolean_index_truth_sequence_start_end_length,
    reduce_repeating_sequences,
    tolerance_modeled_boolean_index_truth_sequence_start_end_length,
)
from bikipy.math.shortcut import np_sum_int
from bikipy.perimeter import RectanglePerimeter
from bikipy.utils.pandas import motion_analysis_indexer_for_subsection
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS
from bikipy.utils.plot.inspect import generic_figure_finalization

logger = getLogger(__name__)

QuadrantGrid = tuple[int, int]


class RectangleEnclosedExperiment(EnclosedExperiment):
    pass


class RectangleEnclosedTrial(EnclosedTrial):
    rectangle_2d_bin: QuadrantGrid = (2, 2)

    trial_perimeter_enclosure_class = RectanglePerimeter

    manual_center_rectangle_dimensions_meters: Optional[Np1DArrayFp64] = None
    center_rectangle_dimensions_to_spatial_resolution_ratio: Optional[float] = None
    center_periphery_tolerance_model: bool = False

    @computed_field  # type: ignore[misc]
    @cached_property
    def _center_periphery_is_defined(self) -> bool:
        return (
            self.manual_center_rectangle_dimensions_meters is not None
            or self.center_rectangle_dimensions_to_spatial_resolution_ratio
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def _inspect_center_periphery_directory(self) -> DirectoryPath:
        result = self.inspection_fig_output_path / "center_periphery"
        result.mkdir(exist_ok=True)
        return result

    @computed_field  # type: ignore[misc]
    @cached_property
    def _inspect_quadrant_directory(self) -> DirectoryPath:
        result = self.inspection_fig_output_path / "quadrant"
        result.mkdir(exist_ok=True)
        return result

    @computed_field  # type: ignore[misc]
    @cached_property
    def quadrant_grid_coordinate_to_vertices(
        self,
    ) -> dict[QuadrantGrid, Np2DArrayFp64]:
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

    @computed_field  # type: ignore[misc]
    @cached_property
    def quadrant_grid_coordinates(self) -> tuple[tuple[int, int], ...]:
        return tuple(self.quadrant_grid_coordinate_to_vertices)

    @computed_field  # type: ignore[misc]
    @cached_property
    def quadrant_index_to_quadrant_grid_coordinate(self) -> dict[int, QuadrantGrid]:
        return {
            i: quadrant_grid_coordinate
            for i, quadrant_grid_coordinate in enumerate(self.quadrant_grid_coordinate_to_vertices, start=1)
        }

    @computed_field  # type: ignore[misc]
    @cached_property
    def quadrant_grid_coordinate_to_quadrant_index(self) -> dict[QuadrantGrid, int]:
        return {
            quadrant_grid_coordinate: i
            for i, quadrant_grid_coordinate in self.quadrant_index_to_quadrant_grid_coordinate.items()
        }

    @computed_field  # type: ignore[misc]
    @cached_property
    def quadrant_grid_coordinate_to_quadrant(self) -> dict[QuadrantGrid, Quadrant]:
        """
        Left to right, top to down
        :return:
        """
        result = {
            quadrant_grid_coordinate: Quadrant(
                vertices_in_meters=self.quadrant_grid_coordinate_to_vertices[quadrant_grid_coordinate],
                kinematic_coordinates=self.reader.kinematic_coordinates,
                fps=self.fps,
                quadrant_index=quadrant_index,
            )
            for quadrant_index, quadrant_grid_coordinate in self.quadrant_index_to_quadrant_grid_coordinate.items()
        }
        if self.inspection_fig_output_path:
            fig, ax = self.video.subplot()
            ax.set_title(f"Quadrants_Trial_#{self.label}")

            coordinates = self.video.prepare_coordinates_for_plotting(self.reader.kinematic_coordinates)

            confinement = np.zeros(self.reader.frames, dtype=bool)

            colors = plt.cm.rainbow(np.linspace(0, 1, len(result) + 1))
            for color, (grid_coordinate, quadrant) in zip(colors, result.items()):
                ax.plot(
                    *quadrant.plot_vertices(self.video).T,
                    label=f"({grid_coordinate[0]}, {grid_coordinate[1]})",
                    color=color,
                )
                ax.scatter(*coordinates[quadrant.confinement_boolean_index].T, color=color)

                confinement = confinement | quadrant.confinement_boolean_index

            ax.scatter(*coordinates[~confinement].T, color=colors[-1], label="Unconfinement")
            ax.scatter(*self.video.center_for_plot.T, color="r", label="VideoCenter")

            if self.manual_center_meters is not None:
                manual_center = self.manual_center_meters
                if self.video.coordinates_need_to_be_scaled_for_plot:
                    manual_center = manual_center * self.video.pixels_per_meter

                ax.scatter(*manual_center.T, color="k", label="ManualCenter")

            plt.legend(**BOTTOM_LEGEND_KWARGS)

            generic_figure_finalization(
                self._inspect_quadrant_directory,
                potential_label=self.label,
                inspect_fig_file_format=INSPECT_FIG_FILE_FORMAT,
            )

        return result

    @computed_field  # type: ignore[misc]
    @cached_property
    def location_sequence_quadrant(self) -> Np2DArrayFp64:
        raw_location_sequence_quadrant = np.zeros(self.number_of_frames, dtype=np.uint8)
        for quadrant_index, quadrant_grid_coordinate in self.quadrant_index_to_quadrant_grid_coordinate.items():
            quadrant = self.quadrant_grid_coordinate_to_quadrant[quadrant_grid_coordinate]

            if np.any(raw_location_sequence_quadrant[quadrant.confinement_boolean_index]):
                logger.warning("Quadrant confinement has temporal-spatial collision with another, ignoring")

            raw_location_sequence_quadrant[quadrant.confinement_boolean_index] = quadrant_index

        return np.array(reduce_repeating_sequences(raw_location_sequence_quadrant, round(self.fps * 0.35)))

    @computed_field  # type: ignore[misc]
    @cached_property
    def quadrant_grid_coordinate_to_entries(self) -> dict[QuadrantGrid, int]:
        result = {
            quadrant_grid_coordinate: np.sum(self.location_sequence_quadrant == quadrant_index)
            for quadrant_index, quadrant_grid_coordinate in self.quadrant_index_to_quadrant_grid_coordinate.items()
        }
        return result

    @computed_field  # type: ignore[misc]
    @cached_property
    def quadrant_grid_coordinate_to_seconds_present(self) -> dict[QuadrantGrid, float]:
        return {
            quadrant_grid_coordinate: quadrant.seconds_present
            for quadrant_grid_coordinate, quadrant in self.quadrant_grid_coordinate_to_quadrant.items()
        }

    @computed_field  # type: ignore[misc]
    @cached_property
    def _center_boolean_index_motion_island(self) -> tuple[TruthIslandMetadata, Np1DArrayBool]:
        raw_center_boolean_index = self.center_rectangle.confinement_boolean_index(
            "rectangle-centre-motion-island",
            self.reader.kinematic_coordinates,
        )
        return tolerance_modeled_boolean_index_truth_sequence_start_end_length(raw_center_boolean_index, self.fps)

    @computed_field  # type: ignore[misc]
    @property
    def center_boolean_index(self) -> Np1DArrayBool:
        return self._center_boolean_index_motion_island[1]

    @computed_field  # type: ignore[misc]
    @property
    def motion_center(self) -> dict[str, float]:
        return merge_motion_island_data(
            self._center_boolean_index_motion_island[0],
            self.reader.kinematic_coordinates,
            self.fps,
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def periphery_boolean_index(self) -> Np1DArrayBool:
        return ~self.center_boolean_index

    @computed_field  # type: ignore[misc]
    @property
    def motion_periphery(self) -> dict[str, float]:
        return merge_motion_island_data(
            boolean_index_truth_sequence_start_end_length(self.periphery_boolean_index),
            self.reader.kinematic_coordinates,
            self.fps,
        )

    @computed_field  # type: ignore[misc]
    @cached_property
    def center_rectangle_dimensions_meters(self) -> Np2DArrayFp64 | None:
        if self._center_periphery_is_defined is None:
            return None
        if self.manual_center_rectangle_dimensions_meters is not None:
            return self.manual_center_rectangle_dimensions_meters
        if self.center_rectangle_dimensions_to_spatial_resolution_ratio is not None:
            return self.metric_resolution / self.center_rectangle_dimensions_to_spatial_resolution_ratio

    @computed_field  # type: ignore[misc]
    @cached_property
    def center_rectangle(self) -> RectanglePerimeter:
        if not self._center_periphery_is_defined:
            msg = (
                "manual_center_rectangle_dimensions_meters or center_rectangle_dimensions_to_spatial_resolution_ratio "
                "must be defined for center_rectangle to be defined"
            )
            raise AttributeError(msg)

        center_point_to_center_rectangle_side_normal_lengths = (
            self.center_rectangle_dimensions_meters * self.video.pixels_per_meter / 2.0
        )

        center = (
            self.video.center_pixels
            if self.manual_center_meters is None
            else self.manual_center_meters * self.video.pixels_per_meter
        )

        # The Y-axis is max at the image origin, hence the inversion WRT the X-axis:
        x_long, y_short = center + center_point_to_center_rectangle_side_normal_lengths
        x_short, y_long = center - center_point_to_center_rectangle_side_normal_lengths

        vertices = np.array(((x_short, y_short), (x_short, y_long), (x_long, y_long), (x_long, y_short)))

        return RectanglePerimeter(vertices_in_pixels=vertices, manual_video=self.video, label="center")

    @computed_field  # type: ignore[misc]
    @cached_property
    def location_sequence_center_periphery(self) -> Np1DArrayUint8:
        # 1 is center, 2 is periphery, 0 is unknown
        location_sequence_center_periphery = np.zeros_like(self.center_boolean_index, dtype=np.uint8)
        location_sequence_center_periphery[self.center_boolean_index] = 1
        location_sequence_center_periphery[self.periphery_boolean_index] = 2
        return np.array(reduce_repeating_sequences(location_sequence_center_periphery, self._frame_tolerance))

    @computed_field  # type: ignore[misc]
    @property
    def center_entries(self) -> int:
        return np_sum_int(self.location_sequence_center_periphery == 1)

    @computed_field  # type: ignore[misc]
    @property
    def periphery_entries(self) -> int:
        return np_sum_int(self.location_sequence_center_periphery == 2)

    @computed_field  # type: ignore[misc]
    @property
    def seconds_on_center(self) -> int:
        return np.sum(self.center_boolean_index) / self.fps

    @computed_field  # type: ignore[misc]
    @property
    def seconds_on_periphery(self) -> int:
        return np.sum(self.periphery_boolean_index) / self.fps

    @computed_field  # type: ignore[misc]
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

        # qgc = Quadrant grid coordinates
        for (qgc_i, quadrant), (qgc_ii, entries) in zip(
            self.quadrant_grid_coordinate_to_quadrant.items(), self.quadrant_grid_coordinate_to_entries.items()
        ):
            assert qgc_i == qgc_ii
            data.extend((*quadrant.motion.values(), entries, quadrant.seconds_present))

        upstream_list.append(
            pd.Series(
                data,
                index=rectangle_motion_column_headers(
                    self.quadrant_grid_coordinates, 2, self._center_periphery_is_defined
                ),
            )
        )

        return upstream_list


class RectangleEnclosedHabituationTrial(RectangleEnclosedTrial, HabituationTrialMixin):
    pass


class BlanketRectangleEnclosedTrial(RectangleEnclosedTrial):
    experiment_class_name = "BlanketRectangleEnclosedExperiment"
    experiment_stage = ExperimentStage.BLANKET


class BlanketRectangleEnclosedExperiment(RectangleEnclosedExperiment):
    experiment_labels = blanket_enclosed_experiment_label_generator("rectangle")

    habituation_trial_class = RectangleEnclosedHabituationTrial
    trial_sequence = (BlanketRectangleEnclosedTrial,)


@lru_cache
def rectangle_motion_column_headers(
    quadrant_grid_coordinates: tuple[tuple[int, int], ...], column_index_levels: int, center_periphery_is_defined: bool
) -> list[tuple[str, ...]]:
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
