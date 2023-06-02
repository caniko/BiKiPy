from functools import cached_property
from typing import ClassVar, Optional, TypeVar

from pydantic import BaseModel, PositiveInt
from schemantic.model.project import SchemanticProjectMixin

from bikipy.core.typing import Label


class BikipyConfigModel(BaseModel):
    class Config:
        underscore_attrs_are_private = True
        keep_untouched = (cached_property,)


class BikipyModel(BikipyConfigModel):
    category: ClassVar[str]


class BikipyHashable(BikipyModel, SchemanticProjectMixin):
    label: Optional[Label]
    int_id: Optional[PositiveInt]

    @classmethod
    @property
    def schemantic_fields_to_exclude_from_config_schema(cls) -> set[str]:
        upstream = super().schemantic_fields_to_exclude_from_config_schema
        upstream.update(("label", "int_id"))
        return upstream

    @property
    def _to_hash(self) -> list:
        return [self.__class__.__name__, self.category, self.int_id, self.label]

    def __hash__(self) -> int:
        return hash(tuple(self._to_hash))

    def __eq__(self, other: "BikipyHashableModel") -> bool:
        try:
            return self._to_hash == other._to_hash
        except AttributeError:
            return False

    def __ne__(self, other: "BikipyHashableModel") -> bool:
        return not self.__eq__(other)


BikipyHashableModel = TypeVar("BikipyHashableModel", bound=BikipyHashable)
