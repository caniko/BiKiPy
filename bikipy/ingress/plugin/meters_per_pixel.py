import numpy as np
from pydantic import DirectoryPath, FilePath, validate_arguments

from bikipy.ingress.plugin.utils import get_file_label_from_3rd_str_in_split
from bikipy.ingress.utils.io import (
    get_perimeter_dir_path,
    initialize_metadata_data_frame,
    load_settings,
)
from bikipy.utils.io.makesense import read_first_makesense_line


def from_makesense_reference_line_segment(data_path: FilePath) -> float:
    meters = float(data_path.stem.split("-")[1])

    segment_tip_a, segment_tip_b = read_first_makesense_line(data_path)
    segment_length = np.linalg.norm(segment_tip_a - segment_tip_b)

    return float(meters / segment_length)


@validate_arguments
def detect_meters_per_pixel_in_perimeter_directory(perimeter_dir: DirectoryPath):
    return {
        get_file_label_from_3rd_str_in_split(meters_per_pixel_file_path): from_makesense_reference_line_segment(
            meters_per_pixel_file_path
        )
        for meters_per_pixel_file_path in perimeter_dir.glob("meters_per_pixel-*.csv")
    }


def detect_global_meters_per_pixel_in_perimeter_directory(perimeter_dir: DirectoryPath):
    return tuple(detect_meters_per_pixel_in_perimeter_directory(perimeter_dir).values())[0]


def validate_metadata_meters_per_pixel_strategy(root_directory: DirectoryPath):
    settings = load_settings(root_directory)
    # perimeter_dir = get_perimeter_dir_path(root_directory)

    metadata = initialize_metadata_data_frame(root_directory, settings["ingress"]["stageful_metadata"])
    if "Meter Pixel Ratio" not in metadata:
        msg = 'Meter Pixel Ratio must be defined in metadata when utilizing the "metadata" strategy'
        raise ValueError(msg)
