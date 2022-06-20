from pydantic import DirectoryPath, FilePath, validate_arguments

from bikipy.utils.io.makesense import image_name_to_point_from_makesense, read_first_makesense_line


def center_file_path_to_value(file_path: FilePath, *args, **kwargs):
    return read_first_makesense_line(file_path)


@validate_arguments
def detect_center_in_perimeter_directory(perimeter_dir: DirectoryPath):
    return {
        _get_center_file_label(center_file_path): image_name_to_point_from_makesense(
            center_file_path, return_first_value=True
        )
        for center_file_path in perimeter_dir.glob("center-*.csv")
    }


@validate_arguments
def _get_center_file_label(center_file_path: FilePath):
    return center_file_path.stem.split("-")[1]
