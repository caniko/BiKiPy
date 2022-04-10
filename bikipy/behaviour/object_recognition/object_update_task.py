from typing import ClassVar

from bikipy.behaviour.object_recognition.base import GenericObjectRecognitionTrial


class ObjectUpdateTaskTraining(GenericObjectRecognitionTrial):
    trial_sequence_index: ClassVar[int] = 1


