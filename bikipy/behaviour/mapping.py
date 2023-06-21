from collections import defaultdict

from pydantic import validate_arguments

from bikipy.behaviour.core.base import ExperimentCLS, TrialCLS
from bikipy.behaviour.core.enclosure.circle import BlanketCircleEnclosedExperiment
from bikipy.behaviour.core.enclosure.rectangle import BlanketRectangleEnclosedExperiment
from bikipy.behaviour.object_recognition.novel_object_recognition import NORTExperiment
from bikipy.behaviour.object_recognition.objects_in_updating_locations import (
    ObjectsInUpdatingLocationsExperiment,
)
from bikipy.behaviour.radial_arm.y_maze import YMazeExperiment
from bikipy.behaviour.reward_tracing.cheeseboard import CheeseboardExperiment

IMPLEMENTED_EXPERIMENTS: set[ExperimentCLS] = {
    NORTExperiment,
    ObjectsInUpdatingLocationsExperiment,
    YMazeExperiment,
    CheeseboardExperiment,
}

GENERIC_EXPERIMENTS: set[ExperimentCLS] = {BlanketCircleEnclosedExperiment, BlanketRectangleEnclosedExperiment}

EXPERIMENTS: set[ExperimentCLS] = {*IMPLEMENTED_EXPERIMENTS, *GENERIC_EXPERIMENTS}

experiment_name_to_class: dict[str, ExperimentCLS] = {}
trial_name_to_trial_class: dict[str, TrialCLS] = {}
experiment_name_to_experiment_stage_to_trial_class: dict[str, dict[str, TrialCLS]] = defaultdict(dict)
for experiment in EXPERIMENTS:
    for label in experiment.experiment_labels:
        experiment_name_to_class[label] = experiment
    experiment_name_to_class[experiment.__name__] = experiment

    for trial_cls in experiment.trial_classes:
        trial_name_to_trial_class[trial_cls.__name__] = trial_cls
        if hasattr(trial_cls, "experiment_stage"):
            experiment_name_to_experiment_stage_to_trial_class[experiment.__name__][
                trial_cls.experiment_stage.value
            ] = trial_cls


@validate_arguments
def resolve_trial(resolver: TrialCLS | str, experiment_name: str) -> TrialCLS:
    assert experiment_name in experiment_name_to_experiment_stage_to_trial_class

    if not isinstance(resolver, str):
        return resolver

    if resolver in trial_name_to_trial_class:
        return trial_name_to_trial_class[resolver]
    if resolver in experiment_name_to_experiment_stage_to_trial_class[experiment_name]:
        return experiment_name_to_experiment_stage_to_trial_class[experiment_name][resolver]

    msg = f"The resolving a Trial class from {resolver} of {experiment_name}"
    raise KeyError(msg)
