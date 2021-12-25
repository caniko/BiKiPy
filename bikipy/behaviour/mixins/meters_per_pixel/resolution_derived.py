"""
Consider using with classes inheriting from VideoMetadataMixin to define
abstract methods
"""
from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from typing import Optional, Union

import numpy as np

from bikipy._base_class import BikipyBase
from bikipy.utils.typing import NDArray


class ResolutionDerivedUnitPerPixelMixin(BikipyBase, ABC):
    metric_resolution: Union[NDArray, float, None] = None
    manual_meters_per_pixel: Optional[float] = None

    @property
    @abstractmethod
    def recording_resolution(self):
        pass

    @property
    def meters_per_pixel(self):
        return self.manual_meters_per_pixel or self.computed_meters_per_pixel

    @cached_property
    def computed_meters_per_pixel(self):
        return _compute_meter_per_pixel(
            self.metric_resolution, self.recording_resolution
        )


class ResolutionDerivedUnitPerPixelTrialMixin(ResolutionDerivedUnitPerPixelMixin, ABC):
    @property
    @abstractmethod
    def _video_metadata_dict_manual_format(self):
        pass

    @cached_property
    def _reader_init_kwargs(self):
        return {
            **self._video_metadata_dict_manual_format,
            **super()._reader_init_kwargs,
        }


@lru_cache
def _compute_meter_per_pixel(
    metric_resolution: Union[np.ndarray, float], recording_resolution: np.ndarray
) -> Union[np.ndarray, float]:
    if not np.any(metric_resolution):
        msg = (
            "metric_resolution attribute needs to be defined to compute "
            "meters_per_pixel"
        )
        raise AttributeError(msg)

    if isinstance(metric_resolution, (float, int)):
        return np.array(metric_resolution) / recording_resolution
    else:
        return metric_resolution / np.mean(recording_resolution)
