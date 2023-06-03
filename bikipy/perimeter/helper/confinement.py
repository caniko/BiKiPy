from logging import getLogger
from typing import Iterable, Sequence

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayFp64

from bikipy._constant import INSPECT_FIG_FILE_FORMAT
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import Perimeter, PerimeterSet
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS
from bikipy.utils.plot.generic import plot_coordinates
from bikipy.utils.plot.inspect import InspectArg, generic_inspection_finalization

logger = getLogger(__file__)


@validate_arguments
def detect_sequential_perimeter_presence(
    coordinates: NDArrayFp64,
    inferior_to_superior_perimeter_instances: Iterable[Perimeter],
    clean_outliers: bool = True,
) -> np.ndarray[bool, bool]:
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
        confinement_boolean_index = single_node_tolerance_model(
            perimeter.confined_coordinate_boolean_index(coordinates), perimeter.video
        )

        if presence[confinement_boolean_index].any():
            overlap_locations[perimeter.label] = np.flatnonzero(presence[confinement_boolean_index])
            presence[overlap_locations[perimeter.label]] = 0
            logger.info(
                f"BaseSinglePerimeter {perimeter.label} has coordinate overlap with "
                f"other perimeter_vertices, {overlap_locations[perimeter.label].size}"
            )

        presence[confinement_boolean_index] = perimeter.int_id

    valid_indices = np.nonzero(presence)
    if clean_outliers:
        presence = presence[valid_indices]

    boolean_array = np.full(coordinates.shape[0], False, dtype=np.bool)
    boolean_array[valid_indices] = True

    return presence, valid_indices, boolean_array


def detect_multi_node_sequential_perimeter_presence(
    coordinates: Sequence[NDArrayFp64],
    inferior_to_superior_perimeter_instances: Sequence[Perimeter],
    inspect_arg: InspectArg = False,
    **inspect_kwargs,
) -> np.ndarray[bool, bool]:
    number_of_perimeters = len(inferior_to_superior_perimeter_instances)

    presence = np.zeros(
        coordinates[0].shape[0],
        dtype=np.uint8 if number_of_perimeters <= 255 else np.uint16,
    )
    overlap_boolean_index = np.zeros(coordinates[0].shape[0], dtype=bool)

    perimeter_id_to_confinement = {}

    for perimeter in inferior_to_superior_perimeter_instances:
        confinement_boolean_index = np.logical_and.reduce(
            [perimeter.confined_coordinate_boolean_index(coordinates) for coordinates in coordinates]
        )
        perimeter_id_to_confinement[perimeter.int_id] = confinement_boolean_index

        if presence[confinement_boolean_index].any():
            overlap_boolean_index = overlap_boolean_index | confinement_boolean_index

        presence[confinement_boolean_index] = perimeter.int_id

    if inspect_arg:
        fig, ax = plt.subplots()

        perimeter_set = PerimeterSet(perimeters=inferior_to_superior_perimeter_instances)

        with sb.color_palette("cubehelix", n_colors=perimeter_set.number_of_vertices):
            perimeter_set.plot(manual_ax=ax)

        has_overlap = np.any(overlap_boolean_index)
        coord_cmap = iter(sb.color_palette("Spectral", n_colors=number_of_perimeters + has_overlap + 1))

        for color, (int_id, label) in zip(coord_cmap, perimeter_set.int_id_to_label.items()):
            if np.any((boolean_index := presence == int_id)):
                plot_coordinates(coordinates[boolean_index], ax, label=label, color=color)

        plot_coordinates(coordinates[~np.any(presence, axis=0)], ax, label="NotConfined", color=next(coord_cmap))

        if has_overlap:
            plot_coordinates(coordinates[overlap_boolean_index], ax, label="Overlap", color=next(coord_cmap))

        ax.legend(**BOTTOM_LEGEND_KWARGS)
        fig.tight_layout()
        generic_inspection_finalization(inspect_arg, f"0-{label}{INSPECT_FIG_FILE_FORMAT}", **inspect_kwargs)

    return presence
