from abc import ABC

from bikipy.behaviour.core.enclosure.rectangle import RectangleEnclosedTrial
from bikipy.behaviour.object_recognition import ObjectRecognitionTrialMixin


class RectangleEnclosedPhysicalObjectTrial(ObjectRecognitionTrialMixin, RectangleEnclosedTrial, ABC):
    pass
