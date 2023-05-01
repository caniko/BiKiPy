from bikipy.behaviour.core.base import HabituationTrialMixin
from bikipy.behaviour.core.enclosure.base import EnclosedExperiment, EnclosedTrial
from bikipy.behaviour.utils import blanket_enclosed_experiment_label_generator
from bikipy.perimeter import CircleFixedRadiusPerimeter


class CircleEnclosedTrial(EnclosedTrial):
    trial_perimeter_enclosure_class = CircleFixedRadiusPerimeter


class CircleEnclosedHabituationTrial(HabituationTrialMixin, CircleEnclosedTrial):
    pass


class CircleEnclosedExperiment(EnclosedExperiment):
    habituation_trial_class = CircleEnclosedHabituationTrial


class BlanketCircleEnclosedExperiment(CircleEnclosedExperiment):
    experiment_labels = blanket_enclosed_experiment_label_generator("circle")
    trial_sequence = (CircleEnclosedTrial,)
