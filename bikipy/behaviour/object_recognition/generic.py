from abc import ABC

from bikipy.behaviour.core.enclosure.rectangle import RectangleEnclosedTrial
from bikipy.behaviour.physical_object import PhysicalObjectTrialMixin


class RectangleEnclosedPhysicalObjectTrial(PhysicalObjectTrialMixin, RectangleEnclosedTrial, ABC):
    pass
