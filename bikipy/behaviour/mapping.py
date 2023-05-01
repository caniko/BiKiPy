from bikipy.behaviour.core.base import ExperimentCLS
from bikipy.behaviour.core.enclosure.circle import BlanketCircleEnclosedExperiment
from bikipy.behaviour.core.enclosure.rectangle import BlanketRectangleEnclosedExperiment
from bikipy.behaviour.object_recognition.novel_object_recognition import NortExperiment
from bikipy.behaviour.object_recognition.objects_in_updating_locations import (
    ObjectsInUpdatingLocationsExperiment,
)
from bikipy.behaviour.radial_arm.y_maze import YMazeExperiment
from bikipy.behaviour.reward_tracing.cheeseboard import CheeseboardExperiment

IMPLEMENTED_EXPERIMENTS: set[ExperimentCLS] = {
    NortExperiment,
    ObjectsInUpdatingLocationsExperiment,
    YMazeExperiment,
    CheeseboardExperiment,
}

GENERIC_EXPERIMENTS: set[ExperimentCLS] = {BlanketCircleEnclosedExperiment, BlanketRectangleEnclosedExperiment}

EXPERIMENTS: set[ExperimentCLS] = {*IMPLEMENTED_EXPERIMENTS, *GENERIC_EXPERIMENTS}

experiment_name_to_class: dict[str, ExperimentCLS] = {}
for experiment in EXPERIMENTS:
    for label in experiment.experiment_labels:
        experiment_name_to_class[label] = experiment
    experiment_name_to_class[experiment.__name__] = experiment
