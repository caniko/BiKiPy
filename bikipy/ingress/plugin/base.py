from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, ClassVar, Optional, TypeVar

import pandas as pd
from pydantic import BaseModel, DirectoryPath, Field, FilePath
from pydantic_numpy.dtype import NDArrayFp64

from bikipy.core.base_class import BikipyModel
from bikipy.core.typing import Label
from bikipy.utils.makesense import get_only_point_from_makesense


class BasePlugin(BikipyModel, ABC):
    ingress: Any = Field(description="Bikipy ingress object to access project metadata relevant for defining perimeter")
    manual_trial_argument_key: Optional[str]

    plural_entries: ClassVar[bool] = False

    ingress_key: ClassVar[str] = ...
    code_key: ClassVar[str] = ...
    default_trial_argument_key: ClassVar[str] = ...

    human_readable_index: ClassVar[str] = ...
    additional_context_columns: ClassVar[Optional[tuple[str]]]

    required: ClassVar[bool] = False
    _inspect: ClassVar[bool] = False

    @cached_property
    def _info(self):
        return self.data_path.stem.split("-")

    @cached_property
    def _plugin_identifier(self) -> list[str, ...]:
        return self._info[0].split(".")

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

    @abstractmethod
    def trialwise_and_metadata(self, trial_id: Label, naive: bool = False):
        ...

    @property
    @abstractmethod
    def globally_defined(self):
        ...


Plugin = TypeVar("Plugin", bound=BasePlugin)


class BasePluginFile(BasePlugin, ABC):
    data_path: FilePath


class BasePluginDirectory(BasePlugin, ABC):
    data_path: DirectoryPath


class HasReferenceMixin(BikipyModel):
    manual_reference: Optional[NDArrayFp64] = Field(
        description="Override the perimeter detection with values defined outside model"
    )

    @cached_property
    def reference_point(self) -> pd.DataFrame | None:
        if self.manual_reference is not None:
            return self.manual_reference
        if (path_to_reference_file := self.data_path.parent / f"reference-{self.label}.csv").exists():
            return get_only_point_from_makesense(path_to_reference_file)


class TrialWiseMetadataOnlyMixin(BaseModel):
    def globally_defined(self) -> None:
        msg = f"{self.__class__.__name__} does not support globally defined"
        raise AttributeError(msg)
