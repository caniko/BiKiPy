from bikipy.behaviour.core import HabituationTrialMixin
from bikipy.behaviour.core.enclosure.base import EnclosedTrial, EnclosedExperiment


class CircleEnclosedTrial(EnclosedTrial):
    pass


class CircleEnclosedHabituationTrial(HabituationTrialMixin, CircleEnclosedTrial):
    pass


class CircleEnclosedExperiment(EnclosedExperiment):
    habituation_trial_class = CircleEnclosedHabituationTrial
