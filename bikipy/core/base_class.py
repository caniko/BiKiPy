from functools import cached_property
from typing import ClassVar, Optional, TypeVar

from pydantic import BaseModel


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
