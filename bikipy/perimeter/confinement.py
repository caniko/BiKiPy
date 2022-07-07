from logging import getLogger
from typing import Sequence, Iterable

import numpy as np
from pydantic import validate_arguments

from bikipy.core.typing import NDArrayFp64, NDArrayBool
from bikipy.feature.tolerance.plural import plural_node_tolerance_filter
from bikipy.feature.tolerance.single import single_node_tolerance_filter
from bikipy.perimeter.base import Perimeter

logger = getLogger(__file__)


@validate_arguments
def detect_sequential_perimeter_presence(
    coordinates: NDArrayFp64,
    inferior_to_superior_perimeter_instances: Iterable[Perimeter],
    clean_outliers: bool = True,
) -> NDArrayBool:
    """

    :param coordinates:
    :param inferior_to_superior_perimeter_instances:
    :param clean_outliers:
    :return:
    """

    presence = np.zeros(
        coordinates.shape[0],
        dtype=np.uint8 if len(inferior_to_superior_perimeter_instances) <= 255 else np.uint16,
    )

    overlap_locations = {}
    for perimeter in inferior_to_superior_perimeter_instances:
        confined_coord_booleans_index = single_node_tolerance_filter(
            perimeter.confined_coordinate_boolean_index(coordinates), perimeter.video
        )

        if presence[confined_coord_booleans_index].any():
            overlap_locations[perimeter.label] = np.flatnonzero(presence[confined_coord_booleans_index])
            presence[overlap_locations[perimeter.label]] = 0
            logger.info(
                f"BaseSinglePerimeter {perimeter.label} has coordinate overlap with "
                f"other perimeter_vertices, {overlap_locations[perimeter.label].size}"
            )

        presence[confined_coord_booleans_index] = perimeter.int_id

    valid_indices = np.nonzero(presence)
    if clean_outliers:
        presence = presence[valid_indices]

    boolean_array = np.full(coordinates.shape[0], False, dtype=np.bool)
    boolean_array[valid_indices] = True

    return presence, valid_indices, boolean_array


def detect_multi_node_sequential_perimeter_presence(
    multi_node_coordinates: Sequence[NDArrayFp64],
    inferior_to_superior_perimeter_instances: Iterable[Perimeter],
    clean_outliers: bool = True,
) -> NDArrayBool:
    presence = np.zeros(
        multi_node_coordinates[0].shape[0],
        dtype=np.uint8 if len(inferior_to_superior_perimeter_instances) <= 255 else np.uint16,
    )

    overlap_locations = {}
    for perimeter in inferior_to_superior_perimeter_instances:
        confined_coord_booleans_index = plural_node_tolerance_filter(
            *(perimeter.confined_coordinate_boolean_index(coordinates) for coordinates in multi_node_coordinates),
            fps=perimeter.video.fps,
        )

        if presence[confined_coord_booleans_index].any():
            overlap_locations[perimeter.label] = np.flatnonzero(presence[confined_coord_booleans_index])
            presence[overlap_locations[perimeter.label]] = 0
            logger.info(
                f"BaseSinglePerimeter {perimeter.label} has coordinate overlap with "
                f"other perimeter_vertices, {overlap_locations[perimeter.label].size}"
            )

        presence[confined_coord_booleans_index] = perimeter.int_id

    valid_indices = np.nonzero(presence)
    if clean_outliers:
        presence = presence[valid_indices]

    boolean_array = np.zeros(multi_node_coordinates[0].shape[0], dtype=np.bool)
    boolean_array[valid_indices] = True

    return presence, valid_indices, boolean_array
