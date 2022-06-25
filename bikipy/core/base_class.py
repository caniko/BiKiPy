from datetime import datetime
from functools import cached_property
from typing import ClassVar, Optional

from pydantic import BaseModel, DirectoryPath, Field


class BaseBikipy(BaseModel):
    class Config:
        underscore_attrs_are_private = True
        keep_untouched = (cached_property,)

    category: ClassVar[Optional[str]]


class BaseBikipyHashable(BaseBikipy):
    int_id: Optional[int]
    label: Optional[str]
    group_label: Optional[str]
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def _hash_key(self):
        return (
            self.best_id,
            self.group_label,
            self.timestamp,
            self.category,
        )

    def __hash__(self):
        return sum(hash(key) for key in self._hash_key)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._hash_key == other._hash_key
        return self._hash_key == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        self._inquire_timestamp_attribute(other)
        return self.timestamp < other.timestamp

    def __le__(self, other):
        self._inquire_timestamp_attribute(other)
        return self.timestamp <= other.timestamp

    def __gt__(self, other):
        self._inquire_timestamp_attribute(other)
        return self.timestamp > other.timestamp

    def __ge__(self, other):
        self._inquire_timestamp_attribute(other)
        return self.timestamp >= other.timestamp

    @property
    def best_id(self):
        return self.label or self.int_id or self.timestamp or self.category

    @staticmethod
    def _inquire_timestamp_attribute(obj):
        if not hasattr(obj, "timestamp"):
            msg = "Cannot perform inequality operations on object without the timestamp attribute"
            raise AttributeError(msg)
