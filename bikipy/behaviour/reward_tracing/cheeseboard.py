from bikipy.behaviour.core.enclosure.circle import (
    CircleEnclosedTrial,
    CircleEnclosedExperiment,
    CircleEnclosedHabituationTrial,
)
from bikipy.behaviour.reward_tracing import RewardTraceTrialMixin


class CheeseboardTrial(RewardTraceTrialMixin, CircleEnclosedTrial):
    trial_label = "cheeseboard_reward_trace"


class CheeseboardExperiment(CircleEnclosedExperiment):
    experiment_labels = {"cheeseboard"}

    habituation_trial_class = CircleEnclosedHabituationTrial
    trial_sequence = (CheeseboardTrial,)
