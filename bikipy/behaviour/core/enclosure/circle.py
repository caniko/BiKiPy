from bikipy.behaviour.core import HabituationTrialMixin
from bikipy.behaviour.core.enclosure.base import EnclosedExperiment, EnclosedTrial
from bikipy.perimeter import CircleFixedRadiusPerimeter


class CircleEnclosedTrial(EnclosedTrial):
    trial_perimeter_enclosure_class = CircleFixedRadiusPerimeter


class CircleEnclosedHabituationTrial(HabituationTrialMixin, CircleEnclosedTrial):
    pass


class CircleEnclosedExperiment(EnclosedExperiment):
    habituation_trial_class = CircleEnclosedHabituationTrial
