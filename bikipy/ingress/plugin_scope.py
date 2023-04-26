from enum import Enum, auto


class PluginScope(str, Enum):
    GLOBAL = "global"
    METADATA = "metadata"
    TRIALWISE = "trialwise"


class ShallowPluginScope(Enum):
    GLOBAL = auto()
    METADATA_TRIALWISE = auto()
