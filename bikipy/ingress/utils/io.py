import os
from functools import lru_cache
from logging import getLogger

import rtoml
import yaml
from pydantic import DirectoryPath, FilePath, validate_arguments

from bikipy import runtime_settings
from bikipy._constant import ANALYSIS_CACHE_STEM_ID
from bikipy.core.typing import Label

logger = getLogger(__name__)

BIKIPY_SETTINGS_FILE_NAME_OLD = "settings.yaml"
BIKIPY_SETTINGS_FILE_NAME = "bikipy_project.toml"


@lru_cache(1)
@validate_arguments
def infer_metadata_path(project_directory: DirectoryPath):
    return next(project_directory.glob("metadata.*"))


@lru_cache(2)
@validate_arguments
def load_settings(project_directory: DirectoryPath, deprecated_file_name: bool = False) -> dict:
    if deprecated_file_name:
        with open(get_project_settings_path(project_directory, True), "r") as in_file:
            return yaml.safe_load(in_file)
    else:
        with open(get_project_settings_path(project_directory), "r") as in_file:
            return rtoml.load(in_file)


def dump_settings(settings_path: FilePath, settings: dict) -> None:
    with open(settings_path, "w") as out_file:
        rtoml.dump(settings, out_file, sort_keys=False)


@lru_cache(2)
@validate_arguments
def get_project_settings_path(project_directory: DirectoryPath) -> FilePath:
    return project_directory / BIKIPY_SETTINGS_FILE_NAME


@lru_cache(1)
@validate_arguments
def get_dataset_directory(project_directory: DirectoryPath) -> DirectoryPath:
    result = project_directory / "dataset"
    result.mkdir(exist_ok=True)
    return result


@lru_cache(1)
@validate_arguments
def get_plugin_directory_path(project_directory: DirectoryPath) -> DirectoryPath:
    result = project_directory / "plugin_files"
    result.mkdir(exist_ok=True)
    return result


@lru_cache(1)
@validate_arguments
def get_inspect_directory_path(project_directory: DirectoryPath) -> DirectoryPath:
    return project_directory / "inspect"


@lru_cache(1)
@validate_arguments
def result_directory_path(project_directory: DirectoryPath) -> DirectoryPath:
    result_directory = project_directory / "result"
    result_directory.mkdir(exist_ok=True)
    return result_directory


def analysis_cache_file_name_from_trial_id(trial_id: Label) -> str:
    return f"{trial_id}_{ANALYSIS_CACHE_STEM_ID}.pickle{runtime_settings.compressed_pickle_suffix}"


@validate_arguments
def flush_analysis_cache(dataset_directory: DirectoryPath) -> None:
    analysis_cache_files = tuple(
        dataset_directory.glob(f"**/*{ANALYSIS_CACHE_STEM_ID}.pickle{runtime_settings.compressed_pickle_suffix}")
    )

    assert analysis_cache_files, "No cache files found"

    joined_file_paths = "\n".join((str(p) for p in analysis_cache_files))
    delete_input = input(f"{joined_file_paths}\nDeleting cache files, confirm (y/N): ")

    if delete_input.lower() != "y":
        import sys

        print("Aborted by user")
        sys.exit(0)

    for f in analysis_cache_files:
        os.remove(f)

    print("Analysis cache deletion complete")
