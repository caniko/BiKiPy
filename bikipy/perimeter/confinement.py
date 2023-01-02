from collections import defaultdict
from functools import partial
from logging import getLogger
from typing import Iterable, Optional, Sequence

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayBool, NDArrayFp64

from bikipy.feature.tolerance.plural import plural_node_tolerance_model
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import Perimeter, PerimeterSet
from bikipy.utils.plot import (
    BOTTOM_LEGEND_KWARGS,
)
from bikipy.utils.plot.generic import plot_coordinates
from bikipy.utils.plot.inspect import InspectArg, generic_inspection_finalization

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
        confined_coord_booleans_index = single_node_tolerance_model(
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
    inspect_arg: InspectArg = False,
    inspect_coords: Optional[NDArrayFp64] = None,
    **inspect_kwargs,
) -> NDArrayBool:
    number_of_perimeters = len(inferior_to_superior_perimeter_instances)

    presence = np.zeros(
        multi_node_coordinates[0].shape[0],
        dtype=np.uint8 if number_of_perimeters <= 255 else np.uint16,
    )

    confinements = {
        perimeter.int_id: [
            perimeter.confined_coordinate_boolean_index(coordinates) for coordinates in multi_node_coordinates
        ]
        for perimeter in inferior_to_superior_perimeter_instances
    }

    int_id_to_overlap_boolean_index = defaultdict(partial(np.zeros_like, presence, dtype=bool))
    for perimeter in inferior_to_superior_perimeter_instances:
        confined_coord_booleans_index = plural_node_tolerance_model(
            *(confinements[perimeter.int_id]),
            fps=perimeter.video.fps,
        )

        if presence[confined_coord_booleans_index].any():
            overlap_boolean_index = confined_coord_booleans_index & presence.astype(bool)
            overlap_boolean_index = overlap_boolean_index & confinements[perimeter.int_id][-1]

            # Remove overlaps that are not on superior node
            overlap_boolean_index[~confinements[perimeter.int_id][-1]] = False
            confined_coord_booleans_index[overlap_boolean_index & ~confinements[perimeter.int_id][-1]] = False
            # Remove the same values from presence
            presence[overlap_boolean_index & ~confinements[perimeter.int_id][-1]] = False

            int_id_to_overlap_boolean_index[perimeter.int_id][overlap_boolean_index] = True

        presence[confined_coord_booleans_index] = perimeter.int_id

    valid_indices = np.nonzero(presence)
    boolean_array = np.zeros_like(presence, dtype=bool)
    boolean_array[valid_indices] = True

    if inspect_arg:
        fig, ax = plt.subplots()

        perimeter_set = PerimeterSet(perimeters=inferior_to_superior_perimeter_instances)

        with sb.color_palette("cubehelix", n_colors=perimeter_set.number_of_vertices):
            perimeter_set.plot(manual_ax=ax)

        coord_cmap = sb.color_palette("Spectral", n_colors=number_of_perimeters + 1)
        for color, (int_id, label) in zip(coord_cmap, perimeter_set.int_id_to_label.items()):
            if np.any((boolean_index := presence == int_id)):
                ax = plot_coordinates(inspect_coords[boolean_index], ax, label=label, color=color)

        ax = plot_coordinates(inspect_coords[~boolean_array], ax, label="NotConfined", color=coord_cmap[-1])

        ax.legend(**BOTTOM_LEGEND_KWARGS)
        fig.tight_layout()
        generic_inspection_finalization(inspect_arg, f"0-{label}.jpg", **inspect_kwargs)

    if clean_outliers:
        presence = presence[valid_indices]

    return presence, valid_indices, boolean_array
