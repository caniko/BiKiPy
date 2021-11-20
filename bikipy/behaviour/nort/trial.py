from functools import cached_property
from logging import getLogger
from typing import Any

from bikipy.behaviour.mixins.misc import OpenFieldTrialMixin, PhysicalObjectTrialMixin
from bikipy.behaviour.square import SquareEnclosedTrial
from bikipy.feature.physical_object import PhysicalObject, PhysicalObjectSet
from bikipy.perimeter.base import Perimeter

logger = getLogger(__name__)


class NortHabituationTrial(SquareEnclosedTrial, OpenFieldTrialMixin):
    """
    NORT experiment without any objects. The purpose of this test is to generate
    reference data for future NORT experiments.
    """

    _trial_sequence_index = 0
    _trial_label = "habituation"


class NortOpenField(NortHabituationTrial):
    pass


class NortTrainingTrial(SquareEnclosedTrial, PhysicalObjectTrialMixin):
    _trial_sequence_index = 1
    _trial_label = "training"
    _feature_summary_column = ("Seconds observing",)

    @cached_property
    def physical_object_variable(self):
        return PhysicalObject(
            self.variable_object_perimeter,
            int_id=1,
            **self._physical_object_keyword_arguments,
        )

    @cached_property
    def physical_object_constant(self):
        return PhysicalObject(
            self.constant_object_perimeter,
            int_id=2,
            **self._physical_object_keyword_arguments,
        )

    @cached_property
    def physical_object_set(self):
        return PhysicalObjectSet(
            (self.physical_object_variable, self.physical_object_constant)
        )

    @property
    def feature_summary_row(self):
        return self.physical_object_set.seconds_observing,


class NortNoveltyTrial(SquareEnclosedTrial, PhysicalObjectTrialMixin):
    novel_object_perimeter: Perimeter
    constant_object_perimeter: Perimeter

    _trial_sequence_index = 2
    _trial_label = "novelty"
    _minimum_seconds_attention = 0.5
    _feature_summary_column = (
        "Absolute discrimination",
        "Discrimination index",
        "Novelty index",
    )

    @cached_property
    def physical_object_novel(self):
        return PhysicalObject(
            self.novel_object_perimeter,
            int_id=1,
            **self._physical_object_keyword_arguments,
        )

    @cached_property
    def physical_object_constant(self):
        return PhysicalObject(
            self.constant_object_perimeter,
            int_id=2,
            **self._physical_object_keyword_arguments,
        )

    @cached_property
    def physical_object_set(self):
        return PhysicalObjectSet(
            (self.physical_object_novel, self.physical_object_constant)
        )

    @cached_property
    def discrimination_index(self):
        return (
            self.physical_object_set.absolute_discrimination / self.experiment_seconds
        )

    @cached_property
    def novelty_preference(self):
        return (
            100.0
            * self.physical_object_novel.attention_filtered_seconds_observing
            / self.experiment_seconds
        )

    @property
    def feature_summary_row(self):
        return (
            self.physical_object_set.absolute_discrimination,
            self.discrimination_index,
            self.novelty_preference,
        )
