from bikipy.behaviour.core.enclosure.circle import (
    CircleEnclosedExperiment,
    CircleEnclosedHabituationTrial,
    CircleEnclosedTrial,
)
from bikipy.behaviour.reward_tracing import RewardTraceTrialMixin
from bikipy.perimeter.circle import CircleVariableRadiusPerimeter
from bikipy.perimeter.polygon.rectangle import RectanglePerimeter


class CheeseboardTrial(RewardTraceTrialMixin[RectanglePerimeter, CircleVariableRadiusPerimeter], CircleEnclosedTrial):
    trial_label = "cheeseboard_reward_trace"


class CheeseboardExperiment(CircleEnclosedExperiment):
    experiment_labels = {"cheeseboard"}

    habituation_trial_class = CircleEnclosedHabituationTrial
    trial_sequence = (CheeseboardTrial,)
