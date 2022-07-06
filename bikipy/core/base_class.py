from functools import cached_property
from typing import ClassVar, Optional, TypeVar

from compress_pickle import compress_pickle
from pydantic import BaseModel, DirectoryPath, Field, root_validator

from bikipy.core.typing import TrialId


class BaseBikipy(BaseModel):
    class Config:
        underscore_attrs_are_private = True
        keep_untouched = (cached_property,)

    category: ClassVar[Optional[str]]


class BaseBikipyHashable(BaseBikipy):
    label: TrialId

    @property
    def _to_hash(self) -> list:
        return [self.__class__.__name__, self.category, self.label]

    def __hash__(self):
        return hash(tuple(self._to_hash))

    def __eq__(self, other: "BikipyHashable"):
        try:
            return self._to_hash == other._to_hash
        except AttributeError:
            return False

    def __ne__(self, other: "BikipyHashable"):
        return not self.__eq__(other)


BikipyHashable = TypeVar("BikipyHashable", bound=BaseBikipyHashable)


class BaseBikipyInspectMixin(BaseBikipy):
    inspect_directory: Optional[DirectoryPath] = Field(description="Path to save figures for inspection of results")
    higher_order_inspect: bool = Field(
        False,
        description="Will only trigger inspection on composite metrics that require the use of several complex functions",
    )
    inspect: bool = Field(False, description="Will trigger all inspection functions in model when True")

    _class_inspect_directory_name: ClassVar[str]

    @root_validator(pre=True)
    def inspect_directory_must_be_defined_when_inspect_is_true(cls, values):
        if "inspect" in values and values["inspect"] and "inspect_directory" not in values:
            msg = "inspect is set to True, yet inspect_directory is None"
            raise AttributeError(msg)
        return values

    @cached_property
    def class_inspect_directory(self) -> DirectoryPath:
        result = self.inspect_directory / self._class_inspect_directory_name
        result.mkdir(exist_ok=True)
        return result

    @cached_property
    def inspect_higher_order(self) -> bool:
        return self.inspect or self.higher_order_inspect

    def save(self):
        compress_pickle.dump(self, self.inspect_directory / f"experiment.pickle.lzma")
