from typing import ClassVar, Optional, TypeVar

from pydantic import BaseModel, NonNegativeInt, computed_field
from schemantic import SchemanticProjectModelMixin

from bikipy.core.typing import Label


class BikipyConfigModel(BaseModel, arbitrary_types_allowed=True):
    pass


class BikipyModel(BikipyConfigModel):
    category: ClassVar[str]


class BikipyHashable(BikipyModel, SchemanticProjectModelMixin):
    label: Optional[Label] = None
    int_id: Optional[NonNegativeInt] = None

    @classmethod
    @property
    def fields_to_exclude_from_single_schema(cls) -> set[str]:
        result = super().fields_to_exclude_from_single_schema
        result.update(("label", "int_id"))
        return result

    @computed_field  # type: ignore[misc]
    @property
    def _to_hash(self) -> list:
        return [self.__class__.__name__, self.category, self.int_id, self.label]

    def __hash__(self) -> int:
        return hash(tuple(self._to_hash))

    def __eq__(self, other) -> bool:
        try:
            return self._to_hash == other._to_hash
        except AttributeError:
            return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
