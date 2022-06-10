import numpy as np
from pydantic import DirectoryPath, FilePath

from bikipy.ingress.utils.io import (
    get_perimeter_dir_path,
    initialize_metadata_data_frame,
    load_settings,
)
from bikipy.utils.io.makesense import read_first_makesense_line


def detect_meter_pixel_ratio(root_directory: DirectoryPath):
    settings = load_settings(root_directory)
    perimeter_dir = get_perimeter_dir_path(root_directory)

    if not settings["meter_pixel_ratio"]:
        msg = "meter_pixel_ratio not set. Either define a strategy for definition, or set a global float value manually"
        raise ValueError(msg)

    match settings["meter_pixel_ratio"]:
        case "global_perimeter":
            meter_pixel_ratio_line_files = tuple(perimeter_dir.glob("meter_pixel_ratio-*.csv"))
            if len(meter_pixel_ratio_line_files) > 1:
                msg = "More than one meter_pixel_ratio line file in perimeter directory"
                raise ValueError(msg)
            if len(meter_pixel_ratio_line_files) != 1:
                msg = 'No perimeter file was found; naming schema: "meter_pixel_ratio-<label>.csv"'
                raise ValueError(msg)

            meter_pixel_ratio_line_file = meter_pixel_ratio_line_files[0]
            return from_makesense_reference_line_segment(
                meter_pixel_ratio_line_file, _meter_from_file_stem(meter_pixel_ratio_line_file)
            )

        case "metadata":
            metadata = initialize_metadata_data_frame(root_directory, settings["ingress"]["stageful_metadata"])
            if "Meter Pixel Ratio" not in metadata:
                msg = 'Meter Pixel Ratio must be defined in metadata when utilizing the "metadata" strategy'
                raise ValueError(msg)


def from_makesense_reference_line_segment(csv_path: FilePath, meters: float) -> float:
    segment_tip_a, segment_tip_b = read_first_makesense_line(csv_path)

    segment_length = np.linalg.norm(segment_tip_a - segment_tip_b)
    return float(meters / segment_length)


def _meter_from_file_stem(file_path: FilePath):
    return float(file_path.stem.split("-")[1])
