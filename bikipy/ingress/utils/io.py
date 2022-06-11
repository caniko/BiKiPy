from functools import lru_cache

import pandas as pd
import yaml
from pydantic import DirectoryPath, FilePath


@lru_cache
def initialize_metadata_data_frame(project_root_directory: DirectoryPath, stageful_metadata: bool):
    return pd.read_excel(
        next(project_root_directory.glob("metadata.*")),
        index_col=0,
        header=(0, 1) if stageful_metadata else 0,
    )


def get_project_settings_path(project_root_directory: DirectoryPath) -> FilePath:
    return project_root_directory / "settings.yaml"


def load_settings(project_root_directory: DirectoryPath) -> dict:
    with open(get_project_settings_path(project_root_directory), "r") as in_file:
        return yaml.safe_load(in_file)


def get_dataset_directory_path(project_root_directory: DirectoryPath) -> DirectoryPath:
    return project_root_directory / "dataset"


def get_perimeter_directory_path(project_root_directory: DirectoryPath) -> DirectoryPath:
    return project_root_directory / "perimeter"
