from functools import cached_property
from typing import ClassVar

from pydantic import DirectoryPath, FilePath

from bikipy.core.base_class import BaseBikipy


class BasePlugin(BaseBikipy):
    data_label: ClassVar[str]

    @cached_property
    def _info(self):
        return self.data_path.stem.split("-")

    @cached_property
    def _plugin_identifier(self) -> list[str, ...]:
        return self._info[0].split(".")

    @property
    def plugin_name(self) -> str:
        return self._plugin_identifier[-1]

    @cached_property
    def sequence_index(self) -> int | None:
        if len(self._plugin_identifier) == 2:
            assert self.plugin_name[0].isdigit()
            return int(self.plugin_name[0])


class BasePluginFile(BasePlugin):
    data_path: FilePath


class BasePluginDirectory(BasePlugin):
    data_path: DirectoryPath
