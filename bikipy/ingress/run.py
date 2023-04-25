from typing import Optional

from projectkit.utils.misc import here_or_there
from pydantic import DirectoryPath

from bikipy import set_bikipy_settings_from_dict
from bikipy.ingress.projectkit import ProjectKitJITBikipyConfiguration


def get_ingress(project_directory: Optional[DirectoryPath] = None):
    project_directory = here_or_there(project_directory)
    return ProjectKitJITBikipyConfiguration.from_config(project_directory).root_class_from_config(
        project_directory=project_directory
    )


def analyze_and_save(project_directory: DirectoryPath):
    ingress = get_ingress(project_directory)

    set_bikipy_settings_from_dict(ingress.project_kit_config["runtime_settings"])

    ingress.save_analysis_data()
