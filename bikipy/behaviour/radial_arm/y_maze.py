from logging import getLogger
from typing import Any, Sequence, Optional, ClassVar

import matplotlib.pyplot as plt
from pydantic import BaseModel

from bikipy.behaviour.radial_arm.base import BaseRadialMazeTrial, \
    BaseRadialMazeExperiment


logger = getLogger(__name__)


class BaseYMaze(BaseModel):
    number_of_arms: ClassVar[Optional[int]] = 3


class YMazeExperiment(BaseRadialMazeExperiment, BaseYMaze):
    pass


class YMazeTrial(BaseRadialMazeTrial, BaseYMaze):
    def plot(
        self,
        ax: Any = None,
        points: Optional[Sequence] = None,
        invalid: bool = False,
    ):
        if points and invalid:
            msg = "points can not be defined while invalid is True"
            raise ValueError(msg)

        if not ax:
            fig, ax = plt.subplots()

        for arm in self.arms:
            arm.plot(ax=ax)

        self.center.plot(
            include_borders=False,
            ax=ax,
            bin=True,
            points=(
                points
                or self.coordinates_per_frame[
                    self.invalid_boolean_index if invalid else self.valid_boolean_index
                ]
            ),
        )

        return ax
