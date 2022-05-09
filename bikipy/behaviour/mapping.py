from typing import Any

from bikipy.behaviour.object_recognition.novel_object_recognition import NortExperiment
from bikipy.behaviour.radial_arm.base import BaseRadialMazeExperiment

NAME_TO_CLASS: dict[str, Any] = {
    "nort": NortExperiment, "NortExperiment": NortExperiment, "radial": BaseRadialMazeExperiment
}
