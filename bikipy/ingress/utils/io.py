from functools import lru_cache

import yaml
from pydantic import DirectoryPath, FilePath, validate_arguments


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
    result.mkdir(exist_ok=True)
    return result
