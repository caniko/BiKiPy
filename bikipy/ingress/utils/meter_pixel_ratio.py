import os.path
from pathlib import Path, PurePath
from typing import Union

import numpy as np
from pydantic import FilePath

from bikipy.utils.io.makesense import from_makesense_line


class MeterPixelRatioNotFoundError(ValueError):
    pass


def get_meter_pixel_ratio(wild_object: Union[int, float, PurePath]):
    if isinstance(wild_object, (int, float)):
        return float(wild_object)
    try:
        wildpath = Path(wild_object).resolve()
    except TypeError:
        raise MeterPixelRatioNotFoundError()
    if os.path.isfile(wildpath):
        return from_makesense_reference_line_segment(wildpath, _meter_from_file_stem(wildpath))
    if os.path.isdir(wildpath) and (perimeter_dir := wildpath / "Perimeter").exists():
        for file_path in perimeter_dir.iterdir():
            if str(file_path.stem).startswith("meter_pixel_ratio-"):
                assert file_path.suffix == ".csv", f"{file_path.suffix} != .csv"
                return from_makesense_reference_line_segment(file_path, _meter_from_file_stem(file_path))
    raise MeterPixelRatioNotFoundError()


def from_makesense_reference_line_segment(csv_path: FilePath, meters: float) -> float:
    segment_tip_a, segment_tip_b = from_makesense_line(csv_path, single_row=True)

    segment_length = np.linalg.norm(segment_tip_a - segment_tip_b)
    return float(meters / segment_length)


def _meter_from_file_stem(file_path: FilePath):
    return float(file_path.stem.split("-")[1])
