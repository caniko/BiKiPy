from functools import cached_property, lru_cache
from typing import ClassVar, Generic, Optional, TypeVar

import numpy as np
from pydantic import validate_arguments
from pydantic_numpy import NDArrayInt16
from skg import ngauss_fit

from bikipy.behaviour.core import BaseExperiment, BaseTrial
from bikipy.perimeter.base import Perimeter


class EnclosedTrial(BaseTrial):
    manual_enclosure: Optional[Perimeter]

    gaussian_dividend_multiplayer: ClassVar[int] = 1

    @property
    def _reader_kwargs(self) -> dict:
        return {**super()._reader_kwargs, "trial_enclosure": self.enclosure}

    @cached_property
    def enclosure(self) -> Perimeter | None:
        return self.manual_enclosure

    @cached_property
    def gaussian_center_to_periphery_score(self) -> float:
        func = gaussian_scoring_field(
            self.video.metric_resolution, gaussian_dividend_multiplayer=self.gaussian_dividend_multiplayer
        )
        scores = np.array(
            [func(*coordinate) for coordinate in self.kinematic_coordinates if not np.any(np.isnan(coordinate))]
        )
        return np.sum(scores) / (self.gaussian_dividend_multiplayer * self.number_of_frames)


class EnclosedExperiment(BaseExperiment):
    pass


@validate_arguments
@lru_cache
def gaussian_scoring_field(resolution: NDArrayInt16, scale: int = 1, gaussian_dividend_multiplayer: int = 1):
    resolution *= scale

    model = ngauss_fit.model(
        x=np.indices(resolution, dtype=float),
        a=gaussian_dividend_multiplayer,
        mu=resolution / 2.0,
        sigma=np.array([[resolution[0] ** 2, 0.0], [0.0, resolution[1] ** 2]]),
        axis=0,
    )

    scale_as_float = float(scale)
    return lambda x, y: model[round(x * scale_as_float)][round(y * scale_as_float)]
