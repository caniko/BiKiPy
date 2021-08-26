from collections import UserDict
from functools import lru_cache
from typing import Iterable, Union, Any


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

    def __init__(self, class_dict: dict, **kwargs):
        self.descending = sorted(class_dict, reverse=True)
        super().__init__(class_dict, **kwargs)

    @staticmethod
    def find_range(sequence: Iterable, value: Union[float, int]):
        for number in sequence:
            if number <= value:
                return number

        msg = f"Provided key is less than the first key in the RangeDict; {value}"
        raise KeyError(msg)

    @lru_cache
    def __getitem__(self, key: Union[float, int]):
        try:
            return super().__getitem__(key)
        except KeyError:
            return super().__getitem__(self.find_range(self.descending, key))

    def __setitem__(self, key: Union[float, int], value: Any):
        if not isinstance(key, (float, int)):
            msg = "Keys in RangeDict(s) have to be either integer or float"
            raise TypeError(msg)

        self.descending.append(key)
        self.descending = sorted(self.descending, reverse=True)

        super().__setitem__(key, value)
