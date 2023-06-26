import os
from logging import getLogger
from pathlib import Path
from typing import Literal, Optional

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
    potential_dir: Optional[str] = None,
    potential_label: Optional[str] = None,
    inspect_fig_file_format: Literal[".svgz", ".jpg"] = INSPECT_FIG_FILE_FORMAT,
) -> None:
    if isinstance(inspect_arg, Path):
        if inspect_arg.suffix:
            file_path = inspect_arg / potential_label if potential_label else inspect_arg

            logger.debug(f"Ensuring that {file_path.parent} directory exists")
            os.makedirs(file_path.parent, exist_ok=True)

            if file_path.stem.split("-")[0].isdigit():
                file_path = int_file_stem_incrementor(file_path)

            file_path = file_path.with_suffix(inspect_fig_file_format)

        else:  # Treated as directory
            if potential_dir:
                inspect_arg = inspect_arg / potential_dir
            logger.debug(f"Creating directory {inspect_arg} for inspection figures")
            os.makedirs(inspect_arg, exist_ok=True)

            stem = f"1-{potential_label}" if potential_label else "1"
            file_path = int_file_stem_incrementor((inspect_arg / stem).with_suffix(inspect_fig_file_format))

        logger.debug(f"Saving inspection figure, file path: {file_path}")
        plt.savefig(file_path)

    else:
        # inspect_arg is most likely a boolean
        if inspect_arg:
            plt.show()

    plt.close()
