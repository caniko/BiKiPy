from collections import UserDict
from typing import Dict, Iterable, SupportsFloat, SupportsInt, Union


def translate_keys(store: Dict, translation: Dict) -> Dict:
    result = {}
    for key, value in store.items():
        result[translation[key]] = value
    return result


def sort_dict_by_key_value(obj: Dict):
    return dict(sorted(obj.items(), key=lambda item: item[0]))


class RangeDict(UserDict):
    """
    Ranges are generated from left to right from keys as the following [left, right).
    Practically speaking, a key in range will return the key referred to as left.

    Useful when working with data that is generalised for a given range of values.
    """

    def __init__(self, class_dict, **kwargs):

        self.descending = sorted(class_dict, reverse=True)

        super().__init__(class_dict, **kwargs)

    @staticmethod
    def find_range(sequence: Iterable, value: Union[SupportsFloat, SupportsInt]):
        for number in sequence:
            if number <= value:
                return number

        msg = f"Provided key is less than the first key in the RangeDict; {value}"
        raise KeyError(msg)

    def __getitem__(self, key: Union[SupportsFloat, SupportsInt]):
        try:
            return super().__getitem__(key)
        except KeyError:
            return super().__getitem__(self.find_range(self.descending, key))

    def __setitem__(self, key, value):
        if not isinstance(key, (int, float)):
            msg = "Keys in RangeDict(s) have to be either integer or float"
            raise TypeError(msg)

        self.descending.append(key)
        self.descending = sorted(self.descending, reverse=True)

        super().__setitem__(key, value)


class ManyToOneDict:
    def __init__(self, class_dict):
        self.next_index = 0
        self._key_to_value_index, self._value_index_to_value = {}, {}
        self.depolymerize_many_to_one(class_dict)

    def depolymerize_many_to_one(self, base_dict: Dict):
        for keys, value in base_dict.items():
            self._value_index_to_value[self.next_index] = value
            for key in keys:
                self._key_to_value_index[key] = self.next_index
            self.next_index += 1





















