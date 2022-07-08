from logging import getLogger
from typing import Any, Optional, Sequence, TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from pydantic import FilePath

from bikipy.core.typing import NDArrayFp64
from bikipy.utils.collection_utils import generic_multi_indexer
from bikipy.utils.image import read_image
from bikipy.utils.makesense import read_makesense_point

if TYPE_CHECKING:
    from bikipy.perimeter.base import Perimeter


logger = getLogger(__file__)


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
    perimeters: Sequence["Perimeter"],
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
