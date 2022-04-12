from typing import Optional

import numpy as np
from pydantic import validate_arguments, FilePath

from bikipy.utils.io.makesense import from_makesense_line


class CentimeterPixelRatio:
    @validate_arguments
    def __init__(self, meter_pixel_ratio: Optional[float], derivation_method: str = "manual", **kwargs):
        self.derivation_method = derivation_method
        self.meter_pixel_ratio = None

        if self.derivation_method == "manual":
            self.meter_pixel_ratio = meter_pixel_ratio
            if not self.meter_pixel_ratio:
                msg = f"meter_pixel_ratio must be provided when the derivation_method is {self.derivation_method}"
                raise AttributeError(msg)
        elif self.derivation_method == "reference_line":
            self.from_makesense_reference_line_segment(**kwargs)
        else:
            msg = f"The {self.derivation_method} is not supported"
            raise NotImplementedError(msg)

    @validate_arguments
    def from_makesense_reference_line_segment(self, csv_path: FilePath, centimeters: float):
        segment_tip_a, segment_tip_b = from_makesense_line(csv_path, single_row=True)

        segment_length = np.linalg.norm(segment_tip_a - segment_tip_b)
        self.meter_pixel_ratio = segment_length / centimeters

    def __repr__(self):
        return self.meter_pixel_ratio
