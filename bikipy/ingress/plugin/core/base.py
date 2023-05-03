from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, ClassVar, Optional, Type, TypeVar

from pydantic import DirectoryPath, FilePath
from schemantic.model.project import SchemanticProjectMixin

from bikipy.core.base import BikipyModel
from bikipy.core.typing import Label
from bikipy.ingress.name_parser import PluginFileStemParse
from bikipy.ingress.plugin.core.plugin_scope import PluginScope


class BasePlugin(BikipyModel, SchemanticProjectMixin, ABC):
    plugin_scope: Optional[PluginScope]
    manual_trial_argument_key: Optional[str]

    plugin_file_stem_parser: ClassVar[Type[PluginFileStemParse]] = PluginFileStemParse

    required: ClassVar[bool] = False
    plural_entries: ClassVar[bool] = False

    ingress_key: ClassVar[str] = ...
    code_key: ClassVar[str] = ...
    default_trial_argument_key: ClassVar[str] = ...
    human_readable_index: ClassVar[str] = ...

    @abstractmethod
    def trialwise_and_metadata(self, trial_id: Label, naive: bool = False):
        ...

    @property
    @abstractmethod
    def globally_defined(self):
        ...

    def _assert_correct_scope_trialwise_metadata(self) -> None:
        assert self.plugin_scope == PluginScope.TRIALWISE or self.plugin_scope == PluginScope.METADATA

    def _assert_correct_scope_global(self) -> None:
        assert self.plugin_scope == PluginScope.GLOBAL

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("ingress", "data_path", "plugin_scope"))
        return upstream

    @cached_property
    def stem_info(self):
        try:
            return self.plugin_file_stem_parser(self.data_path.stem, self.plugin_scope)
        except Exception as e:
            msg = f"There are issues with plugin on {self.data_path}"
            raise ValueError(msg) from e

    @staticmethod
    def _parse_plugin_settings(settings_dict: dict) -> dict[str, Any]:
        return {field: value for field, value in settings_dict.items() if value != ""}

    @property
    def trial_argument_key(self) -> str:
        return self.manual_trial_argument_key or self.default_trial_argument_key

    @property
    def plugin_name(self) -> str:
        return self._plugin_identifier[-1]

    @cached_property
    def sequence_index(self) -> int | None:
        if len(self._plugin_identifier) == 2:
            assert self.plugin_name[0].isdigit()
            return int(self.plugin_name[0])


PluginType = Type[BasePlugin]
Plugin = TypeVar("Plugin", bound=BasePlugin)


class BasePluginFile(BasePlugin, ABC):
    data_path: FilePath


class BasePluginDirectory(BasePlugin, ABC):
    data_path: DirectoryPath
