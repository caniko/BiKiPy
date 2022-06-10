from datetime import date, datetime
from functools import cached_property
from pathlib import Path
from typing import ClassVar, Optional, Union

import compress_pickle
from pydantic import BaseModel, DirectoryPath, Field


class BikipyBase(BaseModel):
    class Config:
        keep_untouched = (cached_property,)

    category: ClassVar[Optional[str]] = None


class BikipyBaseHashable(BikipyBase):
    int_id: Optional[int] = None
    label: Optional[str] = None
    group_label: Optional[str] = None
    timestamp: Union[date, datetime] = Field(default_factory=datetime.now)
    save_root: Optional[DirectoryPath] = None

    def save(self, save_root: Optional[DirectoryPath] = None):
        save_root = Path(save_root or self.save_root)
        assert save_root
        compress_pickle.dump(self, save_root / f"pickle_{self.category}_{self.timestamp}.lzma")

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
