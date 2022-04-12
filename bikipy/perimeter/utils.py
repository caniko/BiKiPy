from typing import Optional, Any, Sequence

import numpy as np
from matplotlib import pyplot as plt
from pydantic import FilePath

from bikipy.utils.misc import read_makesense_point_csv, read_image, generic_multi_indexer
from numpy.typing import NDArray


def distance_between_two_perimeters(perimeter_a, perimeter_b):
    return np.linalg.norm(perimeter_a.centroid - perimeter_b.centroid)


def get_coco_array_from_path_or_array(
    metadata_path: Optional[FilePath] = None,
    coco_array: Optional[NDArray] = None,
):
    msg = "metadata_path and coco_array are defined mutually exclusive"
    if metadata_path and np.any(coco_array):
        raise ValueError(msg)

    if metadata_path:
        result = read_makesense_point_csv(metadata_path)
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
            if isinstance(perimeter.inspect_image, np.ndarray):
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
        perimeter.plot_perimeter(ax=ax, **perimeter_plot_kwargs)

    return ax


def perimeter_multi_indexer(category: Any, level: int):
    return generic_multi_indexer("Seconds present", "Entries")(category, level)
