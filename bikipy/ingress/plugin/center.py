from pydantic import DirectoryPath, FilePath, validate_arguments

from bikipy.utils.io.makesense import get_only_point_from_makesense


def center_file_path_to_value(file_path: FilePath, *args, **kwargs):
    return get_only_point_from_makesense(file_path)


@validate_arguments
def detect_center_in_perimeter_directory(perimeter_dir: DirectoryPath):
    return {
        _get_center_file_label(center_file_path): get_only_point_from_makesense(center_file_path)
        for center_file_path in perimeter_dir.glob("center-*.csv")
    }


@validate_arguments
def _get_center_file_label(center_file_path: FilePath):
    return center_file_path.stem.split("-")[1]
