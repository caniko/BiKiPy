from logging import getLogger
from typing import ClassVar, Optional

from pydantic import BaseModel

from bikipy.behaviour.radial_arm import (
    BaseRadialMazeExperiment,
    BaseRadialMazeTrial,
)

logger = getLogger(__name__)


class BaseYMaze(BaseModel):
    number_of_arms: ClassVar[Optional[int]] = 3


class YMazeTrial(BaseYMaze, BaseRadialMazeTrial):
    trial_label = "Y-Maze"


class YMazeExperiment(BaseYMaze, BaseRadialMazeExperiment):
    trial_sequence = (YMazeTrial,)
