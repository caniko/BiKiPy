from functools import lru_cache
from math import sqrt

import numpy as np
from pydantic import DirectoryPath, FilePath

from bikipy.core.typing import NDArrayFp64
from bikipy.ingress.plugin.utils import get_file_label_from_3rd_str_in_split
from bikipy.ingress.utils.io import initialize_metadata_data_frame, load_settings
from bikipy.utils.io.makesense import read_first_makesense_line


def meters_per_pixel_file_name_to_value(file_path: FilePath, *args, **kwargs):
    return from_makesense_reference_line_segment(file_path)


@lru_cache
def from_makesense_reference_line_segment(data_path: FilePath) -> NDArrayFp64:
    meters = float(data_path.stem.split("-")[1])

    """
    https://www.reddit.com/r/askmath/comments/j1bvfj/getting_catheti_from_hypotenuse_and_catheti_ratio/?utm_source=share&utm_medium=web2x&context=3
    hypotenuse c and the ratio, r, of a and b in a right triangle.

    From a/b=r, you have a=br.

    Plugging in Pythagoras, c2=a2+b2 -> c2=(br)2+b2. Since c and r are known, you can solve for b.
    c2=(1+r2)b2
    b2=c2/(1+r2)
    a2=c2 - b2
    """

    point_i_and_point_ii = np.array(read_first_makesense_line(data_path), dtype=float)
    magnitude_argsort = np.argsort(np.linalg.norm(point_i_and_point_ii, axis=1))
    pixel_a, pixel_b = point_i_and_point_ii[magnitude_argsort]

    pixel_ab_vector = np.abs(pixel_b - pixel_a)
    pixel_x, pixel_y = pixel_ab_vector
    pixel_ab_ratio = np.divide(*pixel_ab_vector)  # a-b intersects on the origin

    meter_y = sqrt(meters**2 / (1 + pixel_ab_ratio))
    meter_x = sqrt(meters**2 - meter_y**2)

    return np.array([meter_x / pixel_x, meter_y / pixel_y])


@lru_cache
def detect_meters_per_pixel_in_perimeter_directory(perimeter_dir: DirectoryPath) -> dict[str, NDArrayFp64]:
    mpr_file_iterator = perimeter_dir.glob("meters_per_pixel-*.csv")
    return {
        get_file_label_from_3rd_str_in_split(meters_per_pixel_file_path): from_makesense_reference_line_segment(
            meters_per_pixel_file_path
        )
        for meters_per_pixel_file_path in mpr_file_iterator
    }


def first_meters_per_pixel_in_perimeter_directory(perimeter_dir: DirectoryPath):
    return detect_meters_per_pixel_in_perimeter_directory(
        next(iter(detect_meters_per_pixel_in_perimeter_directory(perimeter_dir).values()))
    )


def validate_metadata_meters_per_pixel_strategy(project_root_directory: DirectoryPath):
    settings = load_settings(project_root_directory)
    # perimeter_dir = get_plugin_directory_path(project_root_directory)

    metadata = initialize_metadata_data_frame(project_root_directory, settings["ingress"]["stageful_metadata"])
    if "Meter Pixel Ratio" not in metadata:
        msg = 'Meter Pixel Ratio must be defined in metadata when utilizing the "metadata" strategy'
        raise ValueError(msg)
