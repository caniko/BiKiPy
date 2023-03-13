from functools import cached_property
from pathlib import Path
from typing import ClassVar, Optional, TypeVar

from compress_pickle import compress_pickle
from pydantic import BaseModel, DirectoryPath, Field, FilePath
from pydantic_numpy.dtype import NDArrayUint8

from bikipy.core.typing import Label
from bikipy.utils.plot.inspect import InspectArg, inspect_arg_description


class BikipyModel(BaseModel):
    class Config:
        underscore_attrs_are_private = True
        keep_untouched = (cached_property,)

    category: ClassVar[str] = ...

    @classmethod
    @property
    def exclude_from_settings_schema(cls) -> set[str]:
        """
        Some required fields for a class are sometimes highly specific to its respective object. These fields should
        be recorded in this class-property to be excluded by the settings generator function in the ingress module
        :return:
        """
        return set()

    @classmethod
    @property
    def include_from_settings_schema(cls) -> set[str]:
        """
        Some required fields for a class are sometimes highly specific to its respective object. These fields should
        be recorded in this class-property to be excluded by the settings generator function in the ingress module
        :return:
        """
        return set()


class BaseBikipyHashable(BikipyModel):
    label: Optional[Label]
    int_id: Optional[int]

    @classmethod
    @property
    def exclude_from_settings_schema(cls) -> set[str]:
        return super().exclude_from_settings_schema.union({"label", "int_id"})

    @property
    def _to_hash(self) -> list:
        return [self.__class__.__name__, self.category, self.int_id, self.label]

    def __hash__(self):
        return hash(tuple(self._to_hash))

    def __eq__(self, other: "BikipyHashableModel"):
        try:
            return self._to_hash == other._to_hash
        except AttributeError:
            return False

    def __ne__(self, other: "BikipyHashableModel"):
        return not self.__eq__(other)


BikipyHashableModel = TypeVar("BikipyHashableModel", bound=BaseBikipyHashable)


class BaseBikipyInspectMixin(BikipyModel):
    inspect_arg: InspectArg = Field(False, description=inspect_arg_description)
    manual_inspect_image: Optional[NDArrayUint8] = Field(
        description="Image to use as background in the plots for visualising the analysis data",
    )
    inspect_image_path: Optional[FilePath] = Field(
        description="Path to image to use as background in the plots for visualising the analysis data",
    )

    class_inspect_directory_name: ClassVar[Optional[str]] = None

    def save(self, manual_save_path: Optional[DirectoryPath] = None) -> None:
        if manual_save_path:
            save_directory_path = manual_save_path
        elif isinstance(self.inspect_arg, Path):
            save_directory_path = self.inspect_arg
        else:
            msg = "No path provided to save method"
            raise ValueError(msg)

        compress_pickle.dump(self, save_directory_path / "experiment.pickle.lzma")

    @cached_property
    def class_inspect_arg(self) -> InspectArg:
        if isinstance(self.inspect_arg, Path):
            result = self.inspect_arg / (self.class_inspect_directory_name or self.category)
            result.mkdir(exist_ok=True)
            return result

        return self.inspect_arg
