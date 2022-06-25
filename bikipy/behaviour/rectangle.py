from functools import cached_property, lru_cache
from logging import getLogger
from typing import Any, ClassVar, Hashable, Optional

import matplotlib.pyplot as plt
import numpy as np
from pydantic import DirectoryPath, validate_arguments, validator
from pydantic_numpy import NDArray
from skg import ngauss_fit

from bikipy.behaviour.base import BaseExperiment, BaseTrial
from bikipy.behaviour.utils import reduce_repeating_sequences
from bikipy.core.base_class import BaseBikipy
from bikipy.core.typing import NDArrayBool, NDArrayFp64, NDArrayInt16
from bikipy.feature.motion import (
    get_combined_features_from_merged_motion_island_data,
    motion_multi_indexer,
)
from bikipy.perimeter.utils import perimeter_multi_indexer
from bikipy.utils.collection_utils import generic_multi_indexer
from bikipy.utils.math.geometry import clockwise_sort_points
from bikipy.utils.math.point_in_polygon import parallel_point_in_polygon

logger = getLogger(__name__)
quadrant_grid_typing = tuple[int, int]

A = 1
QUADRANT_INSPECTION_DIR_NAME = "PiP_quadrant_location_booleans"
CENTER_INSPECTION_DIR_NAME = "PiP_center_location_booleans"

_TWO_BY_TWO_IN_ENGLISH = {
    "upper_left": (0, 0),
    "upper_right": (0, 1),
    "lower_left": (0, 1),
    "lower_right": (1, 1),
}


def motion_multi_indexer_for_quadrant(category: Any, level: int):
    return generic_multi_indexer(
        "Displacement", "MedianSpeed", "MedianAcceleration", "FreezingTime", "Entries", "SecondsPresent"
    )(category, level)


class Quadrant(BaseBikipy):
    vertices_in_meters: NDArrayFp64
    framewise_confined_coordinates: NDArrayFp64
    fps: float
    quadrant_index: int

    @cached_property
    def confinement_boolean_index(self) -> NDArrayBool:
        return parallel_point_in_polygon(
            self.framewise_confined_coordinates, clockwise_sort_points(self.vertices_in_meters)
        )

    @cached_property
    def seconds_present(self) -> float:
        return np.sum(self.confinement_boolean_index) / self.fps

    @cached_property
    def motion(self) -> dict[str, float]:
        return get_combined_features_from_merged_motion_island_data(
            self.confinement_boolean_index,
            self.framewise_confined_coordinates,
            self.fps,
        )


class RectangleEnclosedExperiment(BaseExperiment):
    center_box_to_spatial_resolution_ratio: ClassVar[Optional[float]] = None
    rectangle_2d_bin: ClassVar[tuple[int, int]] = (2, 2)

    @classmethod
    @property
    def quadrant_grid_coordinates(cls):
        result = []
        for h in range(1, cls.rectangle_2d_bin[0] + 1):
            for v in range(1, cls.rectangle_2d_bin[1] + 1):
                result.append((h, v))
        return result

    @classmethod
    @property
    def motion_column_headers(cls) -> list:
        quadrant_summary_columns = []
        for quadrant_grid_coordinate in cls.quadrant_grid_coordinates:
            category = f"Quadrant{quadrant_grid_coordinate}"
            quadrant_summary_columns.extend(motion_multi_indexer_for_quadrant(category, 2))
        result = [
            *super().motion_column_headers,
            # ["Gaussian", "CenterToPeriphery"],
            *quadrant_summary_columns,
        ]
        if cls.center_box_to_spatial_resolution_ratio:
            result += [
                *motion_multi_indexer("Center", cls.column_index_levels),
                *perimeter_multi_indexer("Center", cls.column_index_levels),
                *motion_multi_indexer("Periphery", cls.column_index_levels),
                *perimeter_multi_indexer("Periphery", cls.column_index_levels),
            ]
        return result

    def trial_keyword_arguments(self, trial_id: Hashable) -> dict:
        result = super().trial_keyword_arguments(trial_id)
        result["rectangle_2d_bin"] = self.rectangle_2d_bin
        result["center_box_to_spatial_resolution_ratio"] = self.center_box_to_spatial_resolution_ratio
        return result


class RectangleEnclosedTrial(BaseTrial):
    inspect_quadrants: bool = False
    rectangle_2d_bin: quadrant_grid_typing = (2, 2)
    center_box_to_spatial_resolution_ratio: Optional[float]

    @validator("inspect_directory")
    def make_categorical_inspection_sub_dirs(cls, value):
        if value and not (quadrant_dir := value / QUADRANT_INSPECTION_DIR_NAME).exists():
            quadrant_dir.mkdir()
            for current_quadrant in (
                "upper_left",
                "upper_right",
                "lower_right",
                "lower_left",
            ):
                (quadrant_dir / current_quadrant).mkdir()
            (value / CENTER_INSPECTION_DIR_NAME).mkdir()
        return value

    @cached_property
    def _quadrant_inspection_dir(self) -> DirectoryPath:
        return self.inspect_directory / QUADRANT_INSPECTION_DIR_NAME

    @cached_property
    def gaussian_center_to_periphery_score(self) -> float:
        func = gaussian_scoring_field(tuple(self.video.metric_resolution))
        scores = np.array(
            [
                func(*coordinate)
                for coordinate in self.framewise_confined_coordinates
                if not np.any(np.isnan(coordinate))
            ]
        )
        return np.sum(scores) / (A * self.number_of_frames)

    @cached_property
    def quadrant_inspect_directory(self) -> DirectoryPath:
        result = self.inspect_directory / "quadrants"
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

                # if self.center_meter_translation is not None:
                #     quadrant += self.center_meter_translation

                result[(h, v)] = quadrant

        return result

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
                framewise_confined_coordinates=self.framewise_confined_coordinates,
                fps=self.video.fps,
                quadrant_index=quadrant_index,
            )
            for quadrant_index, quadrant_grid_coordinate in self.quadrant_index_to_quadrant_grid_coordinate.items()
        }
        if self.inspect:
            fig, ax = plt.subplots()
            ax.set_title(f"Quadrants_Trial_#{self.best_id}")

            confined = np.zeros(self.reader.frames, dtype=bool)

            colors = plt.cm.rainbow(np.linspace(0, 1, len(result) + 1))
            for color, (grid_coordinate, quadrant) in zip(colors, result.items()):
                ax.plot(
                    *quadrant.vertices_in_meters.T, label=f"({grid_coordinate[0]}, {grid_coordinate[1]})", color=color
                )
                ax.scatter(*self.framewise_confined_coordinates[quadrant.confinement_boolean_index].T, color=color)

                confined = confined | quadrant.confinement_boolean_index

            ax.scatter(*self.framewise_confined_coordinates[~confined].T, color=colors[-1], label="Unconfined")
            ax.scatter(*self.video.center_meters.T, color="r", label="Old")
            ax.scatter(*self.manual_center_meters.T, color="k", label="New")

            plt.legend()
            plt.savefig(self.quadrant_inspect_directory / f"{self.best_id}.jpeg")

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
    def center_rectangle_vertices(self) -> NDArrayFp64:
        if self.center_box_to_spatial_resolution_ratio is None:
            msg = "center_box_to_spatial_resolution_ratio must be defined for center and periphery analysis"
            raise AttributeError(msg)

        center_pixel_lengths = self.video.metric_resolution / self.center_box_to_spatial_resolution_ratio
        center_point_to_center_box_side_normal_lengths = center_pixel_lengths / 2.0

        x_short = self.video.center_meters[0] - center_point_to_center_box_side_normal_lengths[0]
        x_long = self.video.center_meters[0] + center_point_to_center_box_side_normal_lengths[0]
        y_short = self.video.center_meters[1] + center_point_to_center_box_side_normal_lengths[1]
        y_long = self.video.center_meters[1] - center_point_to_center_box_side_normal_lengths[1]

        return np.array(((x_short, y_short), (x_short, y_long), (x_long, y_long), (x_long, y_short)))

    @cached_property
    def center_boolean_index(self) -> NDArrayBool:
        return parallel_point_in_polygon(self.framewise_confined_coordinates, self.center_rectangle_vertices)

    @cached_property
    def periphery_boolean_index(self) -> NDArrayBool:
        return ~self.center_boolean_index

    @cached_property
    def motion_center(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.center_boolean_index,
            self.framewise_confined_coordinates,
            self.video.fps,
        )

    @cached_property
    def motion_periphery(self) -> dict:
        return get_combined_features_from_merged_motion_island_data(
            self.periphery_boolean_index,
            self.framewise_confined_coordinates,
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
    def motion_features(self) -> list:
        if self.video.recording_resolution is None:
            return super().motion_features

        quadrant_motion_values = []
        for (qgc_i, quadrant), (qgc_ii, entries) in zip(
            self.quadrant_grid_coordinate_to_quadrant.items(), self.quadrant_grid_coordinate_to_entries.items()
        ):
            assert qgc_i == qgc_ii
            quadrant_motion_values.extend((*quadrant.motion.values(), entries, quadrant.seconds_present))

        result = [
            *super().motion_features,
            # self.gaussian_center_to_periphery_score,
            *quadrant_motion_values,
        ]

        if self.center_box_to_spatial_resolution_ratio:
            result.extend(
                [
                    *self.motion_center.values(),
                    self.center_entries,
                    self.seconds_on_center,
                    *self.motion_periphery.values(),
                    self.periphery_entries,
                    self.seconds_on_periphery,
                ]
            )

        return result


@lru_cache
@validate_arguments
def gaussian_scoring_field(resolution: NDArrayInt16, scale: int = 1):
    resolution *= scale

    model = ngauss_fit.model(
        x=np.indices(resolution, dtype=float),
        a=A,
        mu=resolution / 2.0,
        sigma=np.array([[resolution[0] ** 2, 0.0], [0.0, resolution[1] ** 2]]),
        axis=0,
    )

    scale_as_float = float(scale)
    return lambda x, y: model[round(x * scale_as_float)][round(y * scale_as_float)]
