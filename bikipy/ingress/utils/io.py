from functools import lru_cache

import pandas as pd
import yaml
from pydantic import DirectoryPath, FilePath


@lru_cache
def initialize_metadata_data_frame(root_directory: DirectoryPath, stageful_metadata: bool):
    return pd.read_excel(
        next(root_directory.glob("metadata.*")),
        index_col=0,
        header=(0, 1) if stageful_metadata else 0,
    )


def get_project_settings_path(root_directory: DirectoryPath) -> FilePath:
    return root_directory / "settings.yaml"


def load_settings(root_directory: DirectoryPath) -> dict:
    with open(get_project_settings_path(root_directory), "r") as in_file:
        return yaml.safe_load(in_file)


def get_dataset_dir_path(root_directory: DirectoryPath) -> DirectoryPath:
    return root_directory / "dataset"


def get_perimeter_dir_path(root_directory: DirectoryPath) -> DirectoryPath:
    return root_directory / "perimeter"
