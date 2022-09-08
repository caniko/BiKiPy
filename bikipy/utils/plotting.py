from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import seaborn
from matplotlib import pyplot as plt
from pydantic import validate_arguments
from pydantic_numpy.dtype import NDArrayFp64

from bikipy.utils.misc import int_file_stem_incrementor

if TYPE_CHECKING:
    from bikipy.core.video import VideoMetadata


logger = getLogger(__file__)

InspectArg = Path | bool
inspect_arg_description = (
    "When path to a directory it is used to define the save directory of figures that will be used for inspection"
)


@validate_arguments
def generic_inspection_finalization(
    inspect_arg: InspectArg,
    potential_label: Optional[str] = None,
    debug_save_message: Optional[str] = None,
) -> None:
    try:
        # This should raise a TypeError if it is a bool and not a Path
        file_path = inspect_arg / potential_label if potential_label else inspect_arg

        if not file_path.suffix:
            msg = f"Saving {file_path}: No file-suffix is defined"
            raise ValueError(msg)

        if file_path.stem.split("-")[0].isdigit():
            file_path = int_file_stem_incrementor(file_path)

        plt.savefig(file_path)
        plt.close()
        if debug_save_message:
            logger.debug(debug_save_message)

    except TypeError:
        # inspect_arg is most likely a boolean
        if inspect_arg:
            plt.show()


def plot_coordinates(
    coordinates: NDArrayFp64,
    ax: Any = None,
    inspect_pixels: bool = False,
    video: Optional["VideoMetadata"] = None,
    **plot_kwargs,
):
    from bikipy.core.video import prepare_data_for_plotting

    coordinates = prepare_data_for_plotting(coordinates, inspect_pixels, video)

    if video.image_resize_multiplier:
        coordinates = coordinates * video.image_resize_multiplier

    ax.scatter(*coordinates.T, **plot_kwargs)

    return ax


def make_color_map(n_colors: int):
    return seaborn.color_palette("dark", n_colors)


BOTTOM_LEGEND_KWARGS = {"loc": "upper center", "bbox_to_anchor": (0.5, -0.04), "fancybox": True, "ncol": 3}
