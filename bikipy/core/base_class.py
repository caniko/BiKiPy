from functools import cached_property
from typing import ClassVar, Optional, TypeVar

from compress_pickle import compress_pickle
from pydantic import BaseModel, DirectoryPath, Field, validator


class BaseBikipy(BaseModel):
    class Config:
        underscore_attrs_are_private = True
        keep_untouched = (cached_property,)

    category: ClassVar[Optional[str]]


class BaseBikipyHashable(BaseBikipy):
    label: str

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


class BaseBikipyInspectMixin(BaseModel):
    inspect_directory: Optional[DirectoryPath] = Field(description="Path to save figures for inspection of results")
    inspect: bool = Field(False, description="Will trigger all inspection functions in model when True")

    @validator("inspect_directory", "inspect", pre=True)
    def inspect_directory_must_be_defined_when_inspect_is_true(cls, v):
        inspect_directory, inspect = v
        if inspect and not inspect_directory:
            msg = "inspect is set to True, yet inspect_directory is None"
            raise AttributeError(msg)
        return v

    def save(self):
        compress_pickle.dump(self, self.inspect_directory / f"experiment.pickle.lzma")
