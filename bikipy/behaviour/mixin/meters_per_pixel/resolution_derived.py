"""
Consider using with classes inheriting from VideoMetadataMixin to define
abstract methods
"""
from abc import ABC, abstractmethod
from functools import cached_property, lru_cache
from typing import Optional, Union

import numpy as np

from bikipy.core.base_class import BikipyBase
from bikipy.utils.typing import NDArray


class ResolutionDerivedUnitPerPixelMixin(BikipyBase, ABC):
    metric_resolution: Union[NDArray, float, None] = None
    manual_meters_per_pixel: Optional[float] = None

    @property
    @abstractmethod
    def recording_resolution(self) -> np.ndarray:
        ...

    @cached_property
    def tuple_recording_resolution(self) -> tuple:
        return tuple(self.recording_resolution)

    @property
    def meters_per_pixel(self):
        return self.manual_meters_per_pixel or self.computed_meters_per_pixel

    @cached_property
    def computed_meters_per_pixel(self):
        if not np.any(self.metric_resolution):
            msg = "metric_resolution attribute needs to be defined to compute " "meters_per_pixel"
            raise AttributeError(msg)

        if isinstance(self.metric_resolution, (float, int)):
            return np.array(self.metric_resolution) / self.recording_resolution
        else:
            return self.metric_resolution / np.mean(self.recording_resolution)


class ResolutionDerivedUnitPerPixelTrialMixin(ResolutionDerivedUnitPerPixelMixin, ABC):
    @property
    @abstractmethod
    def _video_metadata_dict_manual_format(self):
        ...

    @cached_property
    def _reader_init_kwargs(self):
        return {
            **self._video_metadata_dict_manual_format,
            **super()._reader_init_kwargs,
        }
