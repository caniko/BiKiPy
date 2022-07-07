from functools import cached_property
from pathlib import Path
from typing import ClassVar, Optional, TypeVar

import cv2
from compress_pickle import compress_pickle
from pydantic import BaseModel, DirectoryPath, Field, root_validator, FilePath

from bikipy.core.typing import TrialId, NDArrayUint8
from bikipy.utils.plotting import InspectArg, inspect_arg_description


class BaseBikipy(BaseModel):
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


class BaseBikipyHashable(BaseBikipy):
    label: Optional[TrialId] = Field(description="")

    @classmethod
    @property
    def exclude_from_settings_schema(cls) -> set[str]:
        return super().exclude_from_settings_schema.union({"label"})

    @property
    def _to_hash(self) -> list:
        return [self.__class__.__name__, self.category, self.label]

    def __hash__(self):
        return hash(tuple(self._to_hash))

    def __eq__(self, other: "BikipyHashable"):
        try:
            return self._to_hash == other._to_hash
        except AttributeError:
            return False

    def __ne__(self, other: "BikipyHashable"):
        return not self.__eq__(other)


BikipyHashable = TypeVar("BikipyHashable", bound=BaseBikipyHashable)


class BaseBikipyInspectMixin(BaseBikipy):
    inspect_arg: InspectArg = Field(False, description=inspect_arg_description)
    manual_inspect_image: Optional[NDArrayUint8] = Field(
        description="Image to use as background in the plots for visualising the analysis data",
    )
    inspect_image_path: Optional[FilePath] = Field(
        description="Path to image to use as background in the plots for visualising the analysis data",
    )

    @root_validator(pre=True)
    def inspect_directory_must_be_defined_when_inspect_is_true(cls, values):
        if "inspect" in values and values["inspect"] and "inspect_directory" not in values:
            msg = "inspect is set to True, yet inspect_directory is None"
            raise AttributeError(msg)
        return values

    @cached_property
    def inspect_image(self):
        return cv2.imread(self.inspect_image_path) if self.inspect_image_path else self.manual_inspect_image

    @cached_property
    def class_inspect_arg(self) -> InspectArg:
        if isinstance(self.inspect_arg, Path):
            assert self.category
            result = self.inspect_arg / self.category
            result.mkdir(exist_ok=True)
            return result
        return bool(self.inspect_arg)

    def save(self, manual_save_path: Optional[DirectoryPath] = None) -> None:
        if manual_save_path:
            save_directory_path = manual_save_path
        elif isinstance(self.inspect_arg, Path):
            save_directory_path = self.inspect_arg
        else:
            msg = "No path provided to save method"
            raise ValueError(msg)

        compress_pickle.dump(self, save_directory_path / f"experiment.pickle.lzma")
