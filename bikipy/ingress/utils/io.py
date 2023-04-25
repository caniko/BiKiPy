import shutil
from functools import lru_cache
from logging import getLogger

import tomlkit
import yaml
from pydantic import DirectoryPath, FilePath, validate_arguments

from bikipy import runtime_settings

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
            return tomlkit.load(in_file)


def dump_settings(settings_path: FilePath, settings: dict) -> None:
    with open(settings_path, "w") as out_file:
        tomlkit.dump(settings, out_file, sort_keys=False)


@lru_cache(2)
@validate_arguments
def get_project_settings_path(project_directory: DirectoryPath, deprecated_file_name: bool = False) -> FilePath:
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
    result = project_directory / "inspect"

    if not runtime_settings.ignore_pre_existing_inspection_directory and result.exists() and tuple(result.glob("**/*")):
        already_exists_prompt = input(
            f"Inspection directory, {result}, already exists. "
            "Proceeding would result in deletion of directory tree. "
            "Would you like to proceed? y/N "
        )
        if already_exists_prompt.strip().lower() != "y":
            import sys

            logger.info("Aborted by user, inspection directory already exists")
            sys.exit(0)

        shutil.rmtree(result)

    result.mkdir(exist_ok=True)
    return result


@lru_cache(1)
@validate_arguments
def result_directory_path(project_directory: DirectoryPath) -> DirectoryPath:
    result_directory = project_directory / "result"
    result_directory.mkdir(exist_ok=True)
    return result_directory
