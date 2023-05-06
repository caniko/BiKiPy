from bikipy.behaviour.core.enclosure.circle import (
    CircleEnclosedExperiment,
    CircleEnclosedHabituationTrial,
    CircleEnclosedTrial,
)
from bikipy.behaviour.reward_tracing import RewardTraceTrialMixin
from bikipy.perimeter.circle.model import CircleVariableRadiusPerimeter
from bikipy.perimeter.polygon.rectangle import RectanglePerimeter


class CheeseboardTrial(RewardTraceTrialMixin[RectanglePerimeter, CircleVariableRadiusPerimeter], CircleEnclosedTrial):
    trial_label = "CheeseboardRewardTrace"

    enclosure_perimeter_object_attribute_names = "start_perimeter"


class CheeseboardExperiment(CircleEnclosedExperiment):
    experiment_labels = {"cheeseboard"}

    habituation_trial_class = CircleEnclosedHabituationTrial
    trial_sequence = (CheeseboardTrial,)
