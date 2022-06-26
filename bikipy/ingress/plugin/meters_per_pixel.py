from functools import cached_property, lru_cache
from logging import getLogger
from math import sqrt
from typing import ClassVar

import numpy as np
from pydantic import DirectoryPath, FilePath, validator

from bikipy.core.typing import NDArrayFp64
from bikipy.ingress.plugin.base import BasePlugin
from bikipy.ingress.utils.io import initialize_metadata_data_frame, load_settings
from bikipy.utils.io.makesense import read_first_makesense_line

logger = getLogger(__file__)


class MeterPerPixel(BasePlugin):
    data_label = "meters_per_pixel"

    @validator("data_path")
    def is_meter_per_pixel_file(cls, value: FilePath):
        match value.stem.split("-")[0].split(".")[-1]:
            case "meter_pixel_ratio":
                logger.warning(
                    f"The {cls.data_label} file has the meter_pixel_ratio indicating it is from an older version"
                )
            case cls.data_label:
                pass
            case _:
                msg = f"Defined file, {value.stem}, is not a {cls.data_label} file"
                raise AttributeError(msg)
        return value

    @cached_property
    def _info(self):
        result = super()._info

        # TODO: Onion validation pydantic v2
        assert len(result) >= 3, (
            f"The file name for {self.data_label} files consist of name, "
            f"method, and meter length delimited by a dash this file: {self.data_path.stem}"
        )

        return result

    @property
    def annotation_method(self) -> str:
        return self._info[1]

    @cached_property
    def meter_length(self) -> float:
        return float(self._info[2])

    @cached_property
    def ratio(self):
        match self.annotation_method:
            case "diagonal":
                """
                We utilize the diagonal of rectangle to derive the components of the two axes on the 2D image.
                We derive both the meters and pixels of the diagonal, and use the Pythagoras theorem for this:

                https://www.reddit.com/r/askmath/comments/j1bvfj/getting_catheti_from_hypotenuse_and_catheti_ratio/?utm_source=share&utm_medium=web2x&context=3
                hypotenuse c and the ratio, r, of a and b in a right triangle.

                From a/b=r, you have a=br.

                Plugging in Pythagoras, c2=a2+b2 -> c2=(br)2+b2. Since c and r are known, you can solve for b.
                c2=(1+r2)b2
                b2=c2/(1+r2)
                a2=c2 - b2
                """

                point_i_and_point_ii = np.array(read_first_makesense_line(self.data_path), dtype=float)
                magnitude_argsort = np.argsort(np.linalg.norm(point_i_and_point_ii, axis=1))
                pixel_a, pixel_b = point_i_and_point_ii[magnitude_argsort]

                pixel_ab_vector = np.abs(pixel_b - pixel_a)
                pixel_x, pixel_y = pixel_ab_vector
                pixel_ab_ratio = np.divide(*pixel_ab_vector)  # a-b intersects on the origin

                meter_y = sqrt(self.meter_length**2 / (1 + pixel_ab_ratio))
                meter_x = sqrt(self.meter_length**2 - meter_y**2)

                return np.array([meter_x / pixel_x, meter_y / pixel_y])
            case "line":
                return self.meter_length / np.array(read_first_makesense_line(self.data_path), dtype=float)
            case _:
                msg = f"Method {self.annotation_method} is not supported"
                raise NotImplementedError(msg)


def meters_per_pixel_file_name_to_value(file_path: FilePath, *args, **kwargs):
    return MeterPerPixel(data_path=file_path).ratio


@lru_cache
def detect_meters_per_pixel_in_perimeter_directory(perimeter_dir: DirectoryPath) -> dict[str, NDArrayFp64]:
    return {
        (mpp := MeterPerPixel(data_path=meters_per_pixel_file_path)).data_label: mpp.ratio
        for meters_per_pixel_file_path in perimeter_dir.glob("meters_per_pixel-*.csv")
    }


def first_meters_per_pixel_in_perimeter_directory(perimeter_dir: DirectoryPath):
    return next(iter(detect_meters_per_pixel_in_perimeter_directory(perimeter_dir).values()))


def validate_metadata_meters_per_pixel_strategy(project_root_directory: DirectoryPath):
    settings = load_settings(project_root_directory)
    # perimeter_dir = get_plugin_directory_path(project_root_directory)

    metadata = initialize_metadata_data_frame(project_root_directory, settings["ingress"]["stageful_metadata"])
    if "Meter Pixel Ratio" not in metadata:
        msg = 'Meter Pixel Ratio must be defined in metadata when utilizing the "metadata" strategy'
        raise ValueError(msg)
