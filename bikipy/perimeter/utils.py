from logging import getLogger
from typing import Any, Optional, Sequence

import numpy as np
from matplotlib import pyplot as plt
from pydantic import FilePath, validate_arguments

from bikipy.core.typing import NDArrayFp64
from bikipy.utils.collection_utils import generic_multi_indexer
from bikipy.utils.image import read_image
from bikipy.utils.io.makesense import read_makesense_point


logger = getLogger(__file__)


@validate_arguments
def detect_sequential_border_presence(
    coordinates: NDArrayFp64,
    superior_poly_border_instances: Sequence,
    inferior_poly_border_instances: Optional[Sequence] = None,
    clean_outliers: bool = True,
):
    """
    Define sequential perimeter confinements of coordinates

    Parameters
    ----------
    coordinates: NDArrayFp64
        Coordinates that will have their confinement tested

    superior_poly_border_instances: Sequence
        PolygonPerimeter instances that will have the highest priority
        in case of overlap with respect to confinement

    inferior_poly_border_instances: Sequence
        PolygonPerimeter instances that will have the lowest priority
        in case of overlap with respect to confinement

    clean_outliers
        Clear elements that aren't confined to any of the given border_vertices
        as a final action before returning the sequential perimeter presence

    Returns
    -------
    NDArrayFp64 that stores the sequential perimeter presence across frames
    """

    perimeter_sequence = (
        superior_poly_border_instances
        if inferior_poly_border_instances is None
        else (*inferior_poly_border_instances, *superior_poly_border_instances)
    )
    presence = np.zeros(
        coordinates.shape[0],
        dtype=np.uint8 if len(perimeter_sequence) <= 255 else np.uint16,
    )

    overlap_locations = {}
    for perimeter in perimeter_sequence:
        confined_coord_booleans_index = perimeter.coordinate_confinement_boolean_index(coordinates)

        if presence[confined_coord_booleans_index].any():
            overlap_locations[perimeter.label] = np.flatnonzero(presence[confined_coord_booleans_index])
            presence[overlap_locations[perimeter.label]] = 0
            logger.info(
                f"BaseSinglePerimeter {perimeter.label} has coordinate overlap with "
                f"other border_vertices, {overlap_locations[perimeter.label].size}"
            )

        presence[confined_coord_booleans_index] = perimeter.int_id

    valid_indices = np.nonzero(presence)
    if clean_outliers:
        presence = presence[valid_indices]

    boolean_array = np.full(coordinates.shape[0], False, dtype=np.bool)
    boolean_array[valid_indices] = True

    return presence, valid_indices, boolean_array


def distance_between_two_perimeters(perimeter_a, perimeter_b):
    return np.linalg.norm(perimeter_a.centroid - perimeter_b.centroid)


def get_coco_array_from_path_or_array(
    metadata_path: Optional[FilePath],
    coco_array: Optional[NDArrayFp64],
):
    msg = "metadata_path and coco_array are defined mutually exclusive"
    if metadata_path and np.any(coco_array):
        raise ValueError(msg)

    if metadata_path:
        result = read_makesense_point(metadata_path)
    elif np.any(coco_array):
        result = coco_array
    else:
        raise ValueError(msg)

    assert np.any(result)
    return result


def plot_perimeters(
    perimeters: Sequence,
    ax: Any = None,
    inspect_image: Any = None,
    perimeter_plot_kwargs: Optional[dict] = None,
):
    if not ax:
        _fig, ax = plt.subplots()

    if inspect_image is None:
        for i, perimeter in enumerate(perimeters):
            if isinstance(perimeter.inspect_image, NDArrayFp64):
                potential_inspect_image = perimeter.inspect_image
                if i == len(perimeters) - 1 or all(
                    perimeter.inspect_image is None or np.all(potential_inspect_image == perimeter.inspect_image)
                    for perimeter in perimeters[i + 1 :]
                ):
                    """
                    Old premature optimisation, DON'T DO THIS AGAIN.
                    Use the found image if and only if it is identical
                    to other inspect_images in the rest of the perimeter objects
                    """
                    inspect_image = potential_inspect_image
                break

    if inspect_image is not None:
        ax.imshow(read_image(inspect_image), cmap="gray", vmin=0, vmax=255)

    perimeter_plot_kwargs = perimeter_plot_kwargs or {}
    for perimeter in perimeters:
        perimeter.plot_perimeter(manual_ax=ax, **perimeter_plot_kwargs)

    return ax


def perimeter_multi_indexer(category: Any, level: int):
    return generic_multi_indexer("SecondsPresent")(category, level)
