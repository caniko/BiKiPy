from abc import ABC

from bikipy.behaviour.core.enclosure.rectangle import RectangleEnclosedTrial
from bikipy.feature.qualia.physical_object_trial_mixin import PhysicalObjectTrialMixin


class RectangleEnclosedPhysicalObjectTrial(PhysicalObjectTrialMixin, RectangleEnclosedTrial, ABC):
    pass
