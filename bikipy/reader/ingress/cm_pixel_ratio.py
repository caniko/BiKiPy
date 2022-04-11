from typing import Optional

import numpy as np
import pandas as pd
from pydantic import validate_arguments, FilePath


class CentimeterPixelRatio:
    @validate_arguments
    def __init__(self, cm_pixel_ratio: Optional[float], derivation_method: str = "manual", **kwargs):
        self.derivation_method = derivation_method
        self.cm_pixel_ratio = None

        if self.derivation_method == "manual":
            self.cm_pixel_ratio = cm_pixel_ratio
            if not self.cm_pixel_ratio:
                msg = f"cm_pixel_ratio must be provided when the derivation_method is {self.derivation_method}"
                raise AttributeError(msg)
        elif self.derivation_method == "reference_line":
            self.from_makesense_reference_line_segment(**kwargs)
        else:
            msg = f"The {self.derivation_method} is not supported"
            raise NotImplementedError(msg)

    @validate_arguments
    def from_makesense_reference_line_segment(self, csv_path: FilePath, centimeters: float):
        coco_data = pd.read_csv(
            csv_path,
            names=("label", "1x", "1y", "2x", "2y", "image_name", "x_res", "y_res"),
        )

        assert len(coco_data) == 1
        segment_tip_a = coco_data.iloc[0].values[1:3].astype(int)
        segment_tip_b = coco_data.iloc[0].values[3:5].astype(int)

        segment_length = np.linalg.norm(np.asarray(segment_tip_a) - np.asarray(segment_tip_b))
        self.cm_pixel_ratio = segment_length / centimeters

    def __repr__(self):
        return self.cm_pixel_ratio
