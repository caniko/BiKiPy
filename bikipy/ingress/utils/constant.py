import yaml
from pydantic import DirectoryPath, FilePath


def get_project_settings_path(root_directory: DirectoryPath) -> FilePath:
    return root_directory / "settings.yaml"


def load_settings(root_directory: DirectoryPath) -> dict:
    with open(get_project_settings_path(root_directory), "r") as in_file:
        return yaml.safe_load(in_file)


def get_perimeter_dir_path(root_directory: DirectoryPath) -> DirectoryPath:
    return root_directory / "Perimeter"


def get_perimeter_pickle_path(root_directory: DirectoryPath, settings: dict) -> FilePath:
    return get_perimeter_dir_path(root_directory) / settings["perimeter_pickle_file"]
