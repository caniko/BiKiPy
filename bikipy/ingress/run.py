from typing import Iterable, Optional

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


def generate_inspection_videos(
    trial_ids: Iterable[Label],
    project_directory: Optional[DirectoryPath] = None,
    output_directory: Optional[DirectoryPath] = None,
    codec: Optional[str] = None,
    **kwargs,
) -> None:
    get_ingress(here_or_there(project_directory)).create_analysis_videos(
        trial_ids, output_directory=output_directory, codec=codec, **kwargs
    )
