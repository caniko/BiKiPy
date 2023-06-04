import os
from logging import getLogger
from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt
from pydantic import validate_arguments

from bikipy._constant import INSPECT_FIG_FILE_FORMAT
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
    if isinstance(inspect_arg, Path):
        if inspect_arg.is_dir():

            def img_name():
                result = f"{current_idx}{potential_label}" if potential_label else str(current_idx)
                return f"{result}{INSPECT_FIG_FILE_FORMAT}"

            logger.debug(f"Ensuring that {inspect_arg} directory exists")
            os.makedirs(inspect_arg, exist_ok=True)

            current_idx = 1
            while (current_path := inspect_arg / img_name()).exists():
                current_idx += 1

            file_path = current_path
        else:
            file_path = inspect_arg / potential_label if potential_label else inspect_arg

            logger.debug(f"Ensuring that {file_path.parent} directory exists")
            os.makedirs(file_path.parent, exist_ok=True)

            if file_path.stem.split("-")[0].isdigit():
                file_path = int_file_stem_incrementor(file_path)

            file_path = file_path.with_suffix(INSPECT_FIG_FILE_FORMAT)

        logger.debug(f"Saving inspection file: {file_path}")
        plt.savefig(file_path)

        if debug_save_message:
            logger.debug(debug_save_message)
    else:
        # inspect_arg is most likely a boolean
        if inspect_arg:
            plt.show()

    plt.close()
