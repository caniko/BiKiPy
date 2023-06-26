from logging import getLogger
from typing import Any, Iterable

import numpy as np
from pydantic import validate_arguments
from pydantic_numpy import NDArray
from pydantic_numpy.dtype import NDArrayFp64

from bikipy.feature.angle import angle_from_a_to_b
from bikipy.perimeter.base import SinglePerimeter

logger = getLogger(__name__)


def ray_direction_filter_circle_triangle(
    perimeter: SinglePerimeter,
    ray_travel_direction_point: NDArrayFp64,
    ray_start_point: NDArrayFp64,
    max_radians: float,
) -> np.ndarray[bool, bool]:
    ray_vectors = ray_travel_direction_point - ray_start_point

    closest_points_on_edges = perimeter.closest_point_on_edge_to_coordinates(ray_travel_direction_point)
    vector_to_closest_point_on_edge = perimeter.vector_to_closest_point_on_edge(ray_travel_direction_point)

    direction_point_is_closer_than_start_point = np.linalg.norm(
        closest_points_on_edges - ray_travel_direction_point, axis=1
    ) <= np.linalg.norm(closest_points_on_edges - ray_start_point, axis=1)

    angle_from_normal_to_ray = angle_from_a_to_b(vector_to_closest_point_on_edge, ray_vectors)

    result = direction_point_is_closer_than_start_point & (np.abs(angle_from_normal_to_ray) <= max_radians)

    return result


def unique_with_counts_zipped(array: NDArray):
    return zip(*np.unique(array, return_counts=True))


@validate_arguments
def exclude_value_from_sequence(sequence: NDArrayFp64, exclude: Any):
    return sequence[sequence != exclude]


def feature_2d_multi_indexer(feature: str, groups: Iterable[str]):
    return [(str(feature), str(group)) for group in groups]


def blanket_experiment_label_generator(experiment_label: str) -> set[str]:
    return {
        experiment_label,
        f"{experiment_label}-enclosed",
        f"blanket-{experiment_label}",
        f"generic-{experiment_label}",
    }


def blanket_enclosed_experiment_label_generator(experiment_label: str) -> set[str]:
    result = blanket_experiment_label_generator(experiment_label)
    result.add(f"{experiment_label.capitalize()}Enclosed")
    return result
