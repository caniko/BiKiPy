from pydantic import DirectoryPath, validate_arguments

from bikipy.reader.ingress.utils.constant import INGRESS_TO_ANALYSIS_FUNCTION
from bikipy.reader.ingress.utils.perimeter import load_settings


def analyse(root_directory: DirectoryPath, *args, **kwargs) -> None:
    settings = load_settings(root_directory)
    return INGRESS_TO_ANALYSIS_FUNCTION[settings["ingress_method"]](*args, **kwargs)
