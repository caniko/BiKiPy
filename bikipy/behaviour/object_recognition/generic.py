from abc import ABC

from bikipy.behaviour.core.enclosure.rectangle import RectangleEnclosedTrial
from bikipy.behaviour.mixins.physical_object import PhysicalObjectTrialMixin


class RectangleEnclosedPhysicalObjectTrial(PhysicalObjectTrialMixin, RectangleEnclosedTrial, ABC):
    pass
