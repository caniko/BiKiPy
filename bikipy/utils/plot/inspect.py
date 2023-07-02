import os
from logging import getLogger
from pathlib import Path
from typing import Literal, Optional

from matplotlib import pyplot as plt
from pydantic import validate_arguments

from bikipy._constant import INSPECT_FIG_FILE_FORMAT
from bikipy.utils.misc import int_file_stem_incrementor

logger = getLogger(__file__)

inspect_arg_description = (
    "When path to a directory it is used to define the save directory of figures that will be used for inspection"
)


@validate_arguments
def generic_inspection_finalization(
    inspection_fig_output_path: Optional[Path],
    potential_dir: Optional[str] = None,
    potential_label: Optional[str] = None,
    inspect_fig_file_format: Literal[".svgz", ".jpg"] = INSPECT_FIG_FILE_FORMAT,
) -> None:
    if isinstance(inspection_fig_output_path, Path):
        if inspection_fig_output_path.suffix:
            file_path = inspection_fig_output_path / potential_label if potential_label else inspection_fig_output_path

            logger.debug(f"Ensuring that {file_path.parent} directory exists")
            os.makedirs(file_path.parent, exist_ok=True)

            if file_path.stem.split("-")[0].isdigit():
                file_path = int_file_stem_incrementor(file_path)

            file_path = file_path.with_suffix(inspect_fig_file_format)

        else:  # Treated as directory
            if potential_dir:
                inspection_fig_output_path = inspection_fig_output_path / potential_dir

            logger.debug(f"Creating directory {inspection_fig_output_path} for inspection figures")
            os.makedirs(inspection_fig_output_path, exist_ok=True)

            stem = f"1-{potential_label}" if potential_label else "1"
            file_path = int_file_stem_incrementor(
                (inspection_fig_output_path / stem).with_suffix(inspect_fig_file_format)
            )

        logger.debug(f"Saving inspection figure, file path: {file_path}")
        plt.savefig(file_path)

    else:
        # inspection_fig_output_path is most likely a boolean
        if inspection_fig_output_path:
            plt.show()

    plt.close()
