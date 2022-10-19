from functools import cached_property, lru_cache
from typing import ClassVar

import numpy as np
from pydantic import validate_arguments
from pydantic_numpy import NDArrayInt16
from skg import ngauss_fit

from bikipy.behaviour.core import BaseTrial, BaseExperiment


class EnclosedTrial(BaseTrial):
    gaussian_dividend_multiplayer: ClassVar[int] = 1

    @cached_property
    def gaussian_center_to_periphery_score(self) -> float:
        @lru_cache
        @validate_arguments
        def gaussian_scoring_field(resolution: NDArrayInt16, scale: int = 1):
            resolution *= scale

            model = ngauss_fit.model(
                x=np.indices(resolution, dtype=float),
                a=self.gaussian_dividend_multiplayer,
                mu=resolution / 2.0,
                sigma=np.array([[resolution[0] ** 2, 0.0], [0.0, resolution[1] ** 2]]),
                axis=0,
            )

            scale_as_float = float(scale)
            return lambda x, y: model[round(x * scale_as_float)][round(y * scale_as_float)]

        func = gaussian_scoring_field(tuple(self.video.metric_resolution))
        scores = np.array(
            [func(*coordinate) for coordinate in self.kinematic_coordinates if not np.any(np.isnan(coordinate))]
        )
        return np.sum(scores) / (self.gaussian_dividend_multiplayer * self.number_of_frames)


class EnclosedExperiment(BaseExperiment):
    pass
