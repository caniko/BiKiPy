from abc import ABC

from bikipy.behaviour.core.enclosure.rectangle import RectangleEnclosedTrial
from bikipy.feature.physical_object.mixin import PhysicalObjectTrialMixin


class RectangleEnclosedPhysicalObjectTrial(PhysicalObjectTrialMixin, RectangleEnclosedTrial, ABC):
    pass
