from logging import getLogger
from typing import Any, Optional, TYPE_CHECKING

from matplotlib import pyplot as plt
from pydantic import validate_arguments, DirectoryPath

from bikipy.core.typing import NDArrayFp64
from bikipy.utils.misc import int_file_stem_incrementor

if TYPE_CHECKING:
    from bikipy.core.video import VideoMetadata


logger = getLogger(__file__)

InspectArg = DirectoryPath | bool
inspect_arg_description = (
    "When path to a directory it is used to define the save directory of figures that will be used for inspection"
)


@validate_arguments
def generic_inspection_finalization(
    inspect_arg: InspectArg,
    potential_label: str,
    debug_save_message: Optional[str] = None,
    function_name: Optional[str] = None,
) -> None:
    try:
        assert inspect_arg.exists()

        if inspect_arg.is_dir():
            if not function_name:
                msg = "function_name must be defined when generic_inspection_finalization is a directory path"
                raise ValueError(msg)
            directory_path = inspect_arg / function_name
            directory_path.mkdir(exist_ok=True)
            inspect_arg = int_file_stem_incrementor(directory_path / f"0-{function_name}.jpg")

        file_path = inspect_arg / potential_label
        if not file_path.suffix:
            msg = f"Saving {potential_label} to {inspect_arg}: No file-suffix is defined"
            raise ValueError(msg)

        plt.savefig(file_path)
        plt.close()
        if debug_save_message:
            logger.debug(debug_save_message)

    except AttributeError:
        # inspect_arg is most likely a boolean
        if inspect_arg:
            plt.show()


def plot_coordinates(
    coordinates: NDArrayFp64, ax: Any = None, inspect_pixels: bool = False, video: Optional["VideoMetadata"] = None
):
    from bikipy.core.video import convert_meters_to_pixels

    if inspect_pixels:
        coordinates = convert_meters_to_pixels(coordinates, video)

    ax.scatter(*coordinates.T)

    return ax
