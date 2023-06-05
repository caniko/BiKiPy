from logging import getLogger
from typing import Iterable, Sequence

import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from pydantic import validate_arguments
from pydantic_numpy import NDArray
from pydantic_numpy.dtype import NDArrayFp64, NDArrayBool

from bikipy._constant import INSPECT_FIG_FILE_FORMAT
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import Perimeter, PerimeterSet
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS
from bikipy.utils.plot.generic import plot_coordinates
from bikipy.utils.plot.inspect import InspectArg, generic_inspection_finalization

logger = getLogger(__file__)


def detect_multi_node_sequential_perimeter_presence(
    coordinates: Sequence[NDArrayFp64], inferior2superior_perimeter_set: PerimeterSet, all_or_false: bool = True
) -> np.ndarray[int, np.dtype[np.uint8] | np.dtype[np.uint16]]:
    """

    :param coordinates:
    :param inferior2superior_perimeter_set:
    :param all_or_false:
    :return:
    """
    presence = np.zeros(
        coordinates[0].shape[0],
        dtype=inferior2superior_perimeter_set.size_respective_dtype,
    )
    overlap_boolean_index = np.zeros(coordinates[0].shape[0], dtype=bool)
    np_logic_func = np.logical_and.reduce if all_or_false else np.logical_or.reduce

    perimeter_id_to_confinement = {}
    for perimeter in inferior2superior_perimeter_set.all_perimeters:
        confinement_boolean_index = np_logic_func(
            [perimeter.confined_coordinate_boolean_index(coordinates) for coordinates in coordinates]
        )
        perimeter_id_to_confinement[perimeter.int_id] = confinement_boolean_index

        if presence[confinement_boolean_index].any():
            overlap_boolean_index = overlap_boolean_index | confinement_boolean_index

        presence[confinement_boolean_index] = perimeter.int_id

    return presence, overlap_boolean_index


def inspect_sequential_confinement(
    inspect_arg: InspectArg,
    perimeter_set: PerimeterSet,
    coordinate: NDArrayFp64,
    presence: NDArray,
    overlap_boolean_index: NDArrayBool,
    **inspect_kwargs,
):
    if not inspect_arg:
        return

    fig, ax = plt.subplots()

    has_overlap = np.any(overlap_boolean_index)
    coord_cmap = iter(sb.color_palette("Spectral", n_colors=perimeter_set.number_of_perimeters + has_overlap + 1))

    for color, (int_id, label) in zip(coord_cmap, perimeter_set.int_id_to_label.items()):
        perimeter_presence = presence == int_id
        if has_overlap:
            perimeter_presence[overlap_boolean_index] = False
        if np.any(perimeter_presence):
            plot_coordinates(coordinate[perimeter_presence], ax, label=label, color=color)

    plot_coordinates(coordinate[~np.any(presence, axis=0)], ax, label="Outside", color=next(coord_cmap))

    if has_overlap:
        plot_coordinates(coordinate[overlap_boolean_index], ax, label="Overlap", color=next(coord_cmap))

    ax.legend(**BOTTOM_LEGEND_KWARGS)
    fig.tight_layout()

    generic_inspection_finalization(inspect_arg, **inspect_kwargs)
