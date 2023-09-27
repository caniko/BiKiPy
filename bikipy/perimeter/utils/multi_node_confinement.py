from logging import getLogger
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import seaborn as sb
from pydantic_numpy.typing import NpNDArray, NpNDArrayBool, NpNDArrayFp64

from bikipy import runtime_settings
from bikipy.core.typing import ConfinementSequence
from bikipy.core.video import VideoMetadata
from bikipy.feature.tolerance.single import single_node_tolerance_model
from bikipy.perimeter.base import PerimeterSet
from bikipy.utils.plot import BOTTOM_LEGEND_KWARGS
from bikipy.utils.plot.generic import plot_coordinates
from bikipy.utils.plot.inspect import generic_inspection_finalization


def detect_multi_node_sequential_perimeter_presence(
    coordinate_set: Sequence[NpNDArrayFp64],
    inferior2superior_perimeter_set: PerimeterSet,
    all_or_false: bool | tuple[bool, ...] = True,
    tolerance_filter: bool = False,
    tolerance_fps: Optional[float] = None,
    tolerance_seconds_attention: float = runtime_settings.minimum_seconds_tolerance,
    tolerance_seconds_distraction: float = runtime_settings.maximum_seconds_distraction,
) -> ConfinementSequence:
    presence = np.zeros(
        coordinate_set[0].shape[0],
        dtype=inferior2superior_perimeter_set.size_respective_dtype,
    )

    np_logic_func = None
    if isinstance(all_or_false, bool):
        np_logic_func = np.logical_and.reduce if all_or_false else np.logical_or.reduce
    elif not isinstance(all_or_false, tuple):
        msg = "all_or_false must be either bool or tuple"
        raise TypeError(msg)

    overlap_boolean_index = np.zeros(coordinate_set[0].shape[0], dtype=bool)
    for iter_idx, perimeter in enumerate(inferior2superior_perimeter_set.all_perimeters):
        if isinstance(all_or_false, tuple):
            np_logic_func = np.logical_and.reduce if all_or_false[iter_idx] else np.logical_or.reduce

        confinement_data = []
        for coordinates in coordinate_set:
            specific_confinement_boolean_index = perimeter.confinement_coordinate_boolean_index(coordinates)
            if tolerance_filter:
                assert tolerance_fps
                confinement_data.append(
                    single_node_tolerance_model(
                        specific_confinement_boolean_index,
                        fps=tolerance_fps,
                        minimum_seconds_attention=tolerance_seconds_attention,
                        maximum_seconds_distraction=tolerance_seconds_distraction,
                    )
                )
            else:
                confinement_data.append(specific_confinement_boolean_index)

        confinement_boolean_index = np_logic_func(confinement_data)

        if presence[confinement_boolean_index].any():
            overlap_boolean_index = overlap_boolean_index | confinement_boolean_index

        presence[confinement_boolean_index] = perimeter.int_id

    return presence, overlap_boolean_index


def inspect_sequential_confinement(
    inspection_fig_output_path: Optional[Path],
    video: VideoMetadata,
    perimeter_set: PerimeterSet,
    coordinates: NpNDArrayFp64,
    presence: NpNDArray,
    overlap_boolean_index: NpNDArrayBool,
    **inspect_kwargs,
):
    if not inspection_fig_output_path:
        return

    fig, ax = video.subplot()

    has_overlap = np.any(overlap_boolean_index)
    coord_cmap = iter(sb.color_palette("Spectral", n_colors=perimeter_set.number_of_perimeters + has_overlap + 2))

    for color, (int_id, label) in zip(coord_cmap, perimeter_set.int_id_to_label.items()):
        perimeter_presence = presence == int_id
        if has_overlap:
            perimeter_presence[overlap_boolean_index] = False
        if np.any(perimeter_presence):
            plot_coordinates(ax, coordinates[perimeter_presence], label=label, color=color)

    plot_coordinates(ax, coordinates[~np.any(presence, axis=0)], label="Outside", color=next(coord_cmap))

    if has_overlap:
        plot_coordinates(ax, coordinates[overlap_boolean_index], label="Overlap", color=next(coord_cmap))

    ax.legend(**BOTTOM_LEGEND_KWARGS)
    fig.tight_layout()

    generic_inspection_finalization(inspection_fig_output_path, **inspect_kwargs)
