from logging import getLogger
from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt
from pydantic import validate_arguments

from bikipy.utils.misc import int_file_stem_incrementor

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
            file_path = file_path.with_suffix(".jpg")

        if file_path.stem.split("-")[0].isdigit():
            file_path = int_file_stem_incrementor(file_path)

        plt.savefig(file_path)

        if debug_save_message:
            logger.debug(debug_save_message)

    except TypeError:
        # inspect_arg is most likely a boolean
        if inspect_arg:
            plt.show()
    finally:
        plt.close()
