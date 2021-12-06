from functools import cached_property
from typing import Union, Optional, Literal

import numpy as np

from bikipy._base_class import VideoMetaDataMixin
from bikipy.utils.typing import NDArray


class ResolutionDerivedUnitPerPixelMixin(VideoMetaDataMixin):
    metric_resolution: Union[NDArray, float, None] = None
    manual_units_per_pixel: Optional[float] = None

    @property
    def units_per_pixel(self):
        return self.manual_units_per_pixel or self.computed_units_per_pixel

    @cached_property
    def computed_units_per_pixel(self):
        if not np.any(self.metric_resolution):
            msg = "metric_resolution attribute needs to be defined to compute units_per_pixel"
            raise AttributeError(msg)

        if isinstance(self.metric_resolution, (float, int)):
            return np.array(self.metric_resolution) / self.recording_resolution
        else:
            return self.metric_resolution / np.mean(self.recording_resolution)
