from collections import UserDict
from functools import lru_cache
from typing import Any

from pydantic import Field, validate_arguments


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

    @validate_arguments
    def __init__(
        self,
        class_dict: dict = Field(default_factory=dict),
        allow_less_than_first_key: int | bool = False,
        allow_greater_than_last_key: int | bool = True,
        **dict_kwargs,
    ):
        super().__init__(class_dict, **dict_kwargs)

        self.ascending = list(sorted(self.data))
        self.descending = self.ascending[::-1]

        self.allow_less_than_first_key = allow_less_than_first_key
        self.allow_greater_than_last_key = allow_greater_than_last_key

    @lru_cache
    def find_key_range(self, value: float):
        for lower_bound, upper_bound in zip(self.descending, self.descending[1:]):
            if lower_bound <= value <= upper_bound:
                return lower_bound

        if self.allow_greater_than_last_key is not False and (
            self.allow_greater_than_last_key is True or self.allow_greater_than_last_key <= value
        ):
            return self.data[self.descending[0]]

        if self.allow_less_than_first_key is not False and (
            self.allow_less_than_first_key is True or self.allow_less_than_first_key >= value
        ):
            return self.data[self.ascending[0]]

        msg = f"Provided key is not defined; {value}"
        raise KeyError(msg)

    def __getitem__(self, key: float):
        try:
            return super().__getitem__(key)
        except KeyError:
            return super().__getitem__(self.find_key_range(key))

    def __setitem__(self, key: float, value: Any):
        if not isinstance(key, float):
            msg = "Keys in RangeDict(s) have to be either integer or float"
            raise TypeError(msg)

        if isinstance(self.allow_less_than_first_key, float):
            assert (
                key > self.allow_less_than_first_key
            ), f"key >= allow_less_than_first_key; {key} >= {self.allow_less_than_first_key}"

        self.descending.append(key)
        self.descending = sorted(self.descending, reverse=True)

        super().__setitem__(key, value)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            type="RangedDict",
            examples=[{5: "foo", 20: "bar"}],
        )

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, cls):
            msg = f"type {type(v)} is not {cls.__name__}"
            raise TypeError(msg)
        return v
