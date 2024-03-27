from typing import Iterable, Optional

from project_kit.utils.misc import here_or_there
from pydantic import DirectoryPath

from bikipy import set_bikipy_settings_from_dict
from bikipy._constant import RUNTIME_SETTINGS_MAP_NAME
from bikipy.core.typing import Label
from bikipy.ingress.projectkit import ProjectKitJITBikipyConfiguration
from bikipy.ingress.workflow.base import BaseIngressWorkflow


def get_ingress(project_directory: Optional[DirectoryPath] = None, **jit_kwargs) -> BaseIngressWorkflow:
    project_directory = here_or_there(project_directory)
    return ProjectKitJITBikipyConfiguration.from_config(project_directory, **jit_kwargs).root_class_from_config(
        project_directory=project_directory
    )


def analyze_and_save(project_directory: DirectoryPath) -> None:
    ingress = get_ingress(project_directory)

    set_bikipy_settings_from_dict(ingress.project_kit_config[RUNTIME_SETTINGS_MAP_NAME])

    ingress.save_analysis_data()


def generate_inspection_videos(
    trial_ids: Iterable[Label],
    project_directory: Optional[DirectoryPath] = None,
    codec: Optional[str] = None,
    **kwargs,
) -> None:
    """

    :param trial_ids:
    :param project_directory:
    :param codec: GPU encoders:
        - Nvidia (<4000-series): hevc_nvenc --new--> "av1_nvenc"
        - AMD: "hevc_amf" --new--> "av1_amf"
        - Intel arc: av1_qsv
    :param kwargs:
    :return:
    """
    ingress = get_ingress(here_or_there(project_directory), root_class_init_kwargs=dict(no_inspection=True))
    ingress.create_analysis_videos(trial_ids, output_directory=ingress.result_directory_path, codec=codec, **kwargs)
