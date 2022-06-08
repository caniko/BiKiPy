from typing import Any

from bikipy.behaviour.object_recognition.novel_object_recognition import NortExperiment
from bikipy.behaviour.object_recognition.objects_in_updating_locations import ObjectsInUpdatingLocationsExperiment
from bikipy.behaviour.radial_arm.base import BaseRadialMazeExperiment


EXPERIMENT_NAME_TO_CLASS: dict[str, Any] = {
    "nort": NortExperiment,
    "novel_object_recognition_test": NortExperiment,
    "NortExperiment": NortExperiment,
    # ---
    "oul": ObjectsInUpdatingLocationsExperiment,
    "objects_in_updating_locations": ObjectsInUpdatingLocationsExperiment,
    "ObjectsInUpdatingLocationsExperiment": ObjectsInUpdatingLocationsExperiment,
    # ---
    "rm": BaseRadialMazeExperiment,
    "radial_maze": BaseRadialMazeExperiment,
    "BaseRadialMazeExperiment": BaseRadialMazeExperiment,
}
