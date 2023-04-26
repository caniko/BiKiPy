from enum import Enum


class PluginScope(str, Enum):
    GLOBAL = "global"
    METADATA = "metadata"
    TRIALWISE = "trialwise"
