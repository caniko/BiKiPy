import shutil
from functools import lru_cache
from logging import getLogger

import yaml
from pydantic import DirectoryPath, FilePath, validate_arguments

from bikipy import runtime_settings

logger = getLogger(__name__)


@lru_cache(1)
@validate_arguments
def infer_metadata_path(project_root_directory: DirectoryPath):
    return next(project_root_directory.glob("metadata.*"))


@lru_cache(1)
@validate_arguments
def load_settings(project_root_directory: DirectoryPath) -> dict:
    with open(get_project_settings_path(project_root_directory), "r") as in_file:
        return yaml.safe_load(in_file)


@lru_cache(1)
@validate_arguments
def get_project_settings_path(project_root_directory: DirectoryPath) -> FilePath:
    return project_root_directory / "settings.yaml"


@lru_cache(1)
@validate_arguments
def get_dataset_directory_path(project_root_directory: DirectoryPath) -> DirectoryPath:
    result = project_root_directory / "dataset"
    result.mkdir(exist_ok=True)
    return result


@lru_cache(1)
@validate_arguments
def get_plugin_directory_path(project_root_directory: DirectoryPath) -> DirectoryPath:
    result = project_root_directory / "plugin_files"
    result.mkdir(exist_ok=True)
    return result


@lru_cache(1)
@validate_arguments
def get_inspect_directory_path(project_root_directory: DirectoryPath) -> DirectoryPath:
    result = project_root_directory / "inspect"

    if not runtime_settings.ignore_pre_existing_inspection_directory and result.exists():
        already_exists_prompt = input(
            f"Inspection directory, {result}, already exists."
            f"Proceeding would result in deletion of directory tree. Would you like to proceed? y/N"
        )
        if already_exists_prompt.strip().lower() != "y":
            import sys

            logger.info("Aborted by user, inspection directory already exists")
            sys.exit(0)

        shutil.rmtree(result)

    result.mkdir(exist_ok=True)
    return result
