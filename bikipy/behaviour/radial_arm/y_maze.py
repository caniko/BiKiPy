from logging import getLogger
from typing import Any, ClassVar, Optional

import matplotlib.pyplot as plt
from pydantic import BaseModel

from bikipy.behaviour.radial_arm.base import (
    BaseRadialMazeExperiment,
    BaseRadialMazeTrial,
)
from bikipy.core.typing import NDArrayFp64

logger = getLogger(__name__)


class BaseYMaze(BaseModel):
    number_of_arms: ClassVar[Optional[int]] = 3


class YMazeTrial(BaseYMaze, BaseRadialMazeTrial):
    def plot(
        self,
        ax: Any = None,
        coordinates: Optional[NDArrayFp64] = None,
        inspect_pixels: bool = False,
        invalid: bool = False,
    ):
        if coordinates and invalid:
            msg = "coordinates can not be defined while invalid is True"
            raise ValueError(msg)

        if not ax:
            fig, ax = plt.subplots()

        for arm in self.arms:
            arm.plot(ax=ax)

        self.center.plot(
            include_borders=False,
            ax=ax,
            inspect_pixels=inspect_pixels,
            bin=True,
            coordinates=(
                coordinates
                or self.framewise_confined_coordinates[
                    self.invalid_boolean_index if invalid else self.valid_boolean_index
                ]
            ),
        )

        return ax


class YMazeExperiment(BaseYMaze, BaseRadialMazeExperiment):
    trial_class: ClassVar[Any] = YMazeTrial
