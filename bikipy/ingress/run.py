from pydantic import DirectoryPath

from bikipy.ingress.utils.settings import auto_define_ingress_object
from bikipy.ingress.workflow import INGRESS_METHOD_NAME_TO_INGRESS_CLASS


def from_project_directory(project_directory: DirectoryPath):
    kwargs = {"project_directory": project_directory}
    try:
        return INGRESS_METHOD_NAME_TO_INGRESS_CLASS[auto_define_ingress_object(project_directory).ingress_method](
            **kwargs
        )
    except KeyError:
        msg = (
            f"Defined ingress method, {auto_define_ingress_object(project_directory).ingress_method}, "
            f"is not supported"
        )
        raise AttributeError(msg)


def analyze_and_save(project_directory: DirectoryPath):
    from_project_directory(project_directory).save_analysis_data()
