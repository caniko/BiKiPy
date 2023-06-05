from abc import ABC, abstractmethod
from collections import deque

from bikipy.ingress.plugin.core.plugin_scope import PluginScope


class PluginFileStemParse(ABC):
    def __init__(self, stem: str, plugin_scope: PluginScope):
        self.stage: str | None

        if plugin_scope == PluginScope.TRIALWISE:
            self.stage, info = stem.split(".")
        else:
            self.stage = None
            info = stem

        self.stem = stem

        self.split = deque(info.split("-"))
        self.identifier = self.split.popleft()

        self.__pop_split_till_empty__()

        assert not self.split, (
            f"The plugin data file, {stem}, has incorrect formatting; "
            f"there are unresolved stem attributes: {self.split}"
        )
        del self.split

    @abstractmethod
    def __pop_split_till_empty__(self) -> None:
        ...


class PluginFileStemParseLastIsLabel(PluginFileStemParse):
    def __pop_split_till_empty__(self) -> None:
        self.label = self.split.pop()
