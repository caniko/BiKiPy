import datetime
from pathlib import Path
from typing import Any

import compress_pickle

from bikipy.utils.typing import Path_typing_kwarg


class BikipyBase:
    category = None

    def __init__(
        self,
        label: Any = None,
        timestamp: Any = None,
        save_root: Path_typing_kwarg = None,
    ):
        self.label = label
        self.timestamp = timestamp or datetime.datetime.now()
        self.save_root = Path(save_root) if save_root else None

    def save(self, save_root: Path_typing_kwarg = None):
        save_root = Path(save_root or self.save_root)
        assert save_root
        compress_pickle.dump(
            self, save_root / f"pickle_{self.category}_{self.timestamp}.lzma"
        )

    @property
    def _hash_key(self):
        raise NotImplementedError

    def __hash__(self):
        return sum(hash(key) for key in self._hash_key)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._hash_key == other._hash_key
        return self._hash_key == other

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def _inquire_timestamp_attribute(obj):
        if not hasattr(obj, "timestamp"):
            msg = "Cannot perform inequality operations on object without the timestamp attribute"
            raise AttributeError(msg)

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
