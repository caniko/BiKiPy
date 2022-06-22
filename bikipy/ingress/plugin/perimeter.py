from functools import lru_cache
from logging import getLogger

import pandas as pd
from pydantic import DirectoryPath, FilePath, PositiveInt

from bikipy.ingress.utils.io import (
    infer_metadata_path,
    load_settings,
)

logger = getLogger(__name__)


def perimeter_file_path_to_value(file_path: FilePath, trial_id: str | PositiveInt, ingress, *args, **kwargs):
    return ingress.first_perimeter_set_from_makesense(file_path, trial_id)


def get_perimeter_data(perimeter_path: FilePath):
    split_file_stem = perimeter_path.stem.split("-")
    assert split_file_stem[0].lower().endswith("perimeter")
    assert len(split_file_stem) == 3
    # return {"shape": split_file_stem[1], "label": split_file_stem[2]}
    return split_file_stem[1:]


@lru_cache(1)
def get_perimeter_name_df(project_root_directory: DirectoryPath):
    assert load_settings(project_root_directory)["ingress"]["perimeter_naming_strategy"] == "metadata"
    df = pd.read_excel(infer_metadata_path(project_root_directory), sheet_name="perimeter_label", index_col=0)
    return df


def get_name_map_from_name_df(project_root_directory: DirectoryPath, trial_id: str | int) -> dict[str, str]:
    return get_perimeter_name_df(project_root_directory)[trial_id]
