from typing import Optional, Iterable

from projectkit.utils.misc import here_or_there
from pydantic import DirectoryPath

from bikipy import set_bikipy_settings_from_dict
from bikipy._constant import RUNTIME_SETTINGS_MAP_NAME
from bikipy.core.typing import Label
from bikipy.ingress.projectkit import ProjectKitJITBikipyConfiguration


def get_ingress(project_directory: Optional[DirectoryPath] = None):
    project_directory = here_or_there(project_directory)
    return ProjectKitJITBikipyConfiguration.from_config(project_directory).root_class_from_config(
        project_directory=project_directory
    )


def analyze_and_save(project_directory: DirectoryPath) -> None:
    ingress = get_ingress(project_directory)

    set_bikipy_settings_from_dict(ingress.project_kit_config[RUNTIME_SETTINGS_MAP_NAME])

    ingress.save_analysis_data()


def generate_inspection_videos(project_directory: DirectoryPath, trial_ids: Iterable[Label]) -> None:
    ingress = get_ingress(project_directory)
    ingress.trial_ids_to_analyse = trial_ids
