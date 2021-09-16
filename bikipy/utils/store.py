from collections import UserDict
from functools import cached_property, lru_cache
from typing import Any, Iterable, Union


def translate_keys(store: dict, translation: dict) -> dict:
    result = {}
    for key, value in store.items():
        result[translation[key]] = value
    return result


def sort_dict_by_key_value(obj: dict):
    return dict(sorted(obj.items(), key=lambda item: item[0]))


class RangeDict(UserDict):
    """
    Ranges are generated from left to right from keys as the following [left, right).
    Practically speaking, a key in range will return the key referred to as left.

    Useful when working with data that is generalised for a given range of values.
    """

    def __init__(
        self,
        class_dict: Union[dict, None] = None,
        allow_less_than_first_key: Union[float, int, bool] = False,
        **kwargs,
    ):
        if not isinstance(allow_less_than_first_key, (bool, int, float)):
            msg = "allow_less_than_first_key can either be bool, int, or float"
            raise TypeError(msg)

        self.descending = sorted(class_dict, reverse=True) if class_dict else {}
        self.allow_less_than_first_key = allow_less_than_first_key

        super().__init__(class_dict, **kwargs)

    def find_key_range(self, value: Union[float, int]):
        for number in self.descending:
            if number <= value:
                return number

        if self.allow_less_than_first_key is not False and (
            # must be value < self._smallest_key
            self.allow_less_than_first_key is True
            or self.allow_less_than_first_key <= value
        ):
            return self._smallest_key

        msg = f"Provided key is less than the first key in the RangeDict; {value}"
        raise KeyError(msg)

    @lru_cache
    def __getitem__(self, key: Union[float, int]):
        try:
            return super().__getitem__(key)
        except KeyError:
            return super().__getitem__(self.find_key_range(key))

    def __setitem__(self, key: Union[float, int], value: Any):
        if not isinstance(key, (float, int)):
            msg = "Keys in RangeDict(s) have to be either integer or float"
            raise TypeError(msg)

        if isinstance(self.allow_less_than_first_key, (int, float)):
            assert (
                key > self.allow_less_than_first_key
            ), f"key >= allow_less_than_first_key; {key} >= {self.allow_less_than_first_key}"

        self.descending.append(key)
        self.descending = sorted(self.descending, reverse=True)

        super().__setitem__(key, value)

    @cached_property
    def _smallest_key(self):
        return self.descending[-1]

    def __hash__(self):
        return 0
