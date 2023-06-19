from enum import Enum


class ExperimentStage(Enum):
    SINGLE = "single"
    BLANKET = "blanket"

    HABITUATION = "habituation"
    TRAINING = "training"
    UPDATE = "update"
    TEST = "test"
