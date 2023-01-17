from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, ClassVar, Optional, Literal

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic_numpy.dtype import NDArrayFp64

from bikipy.behaviour.core.abstract import AbstractTrial
from bikipy.feature.physical_object import ANIMAL_LABEL_TO_PHYSICAL_OBJECT_SET_CLASS
from bikipy.feature.physical_object.set import PhysicalObjectSetCLS, PhysicalObjectSet
from bikipy.perimeter.base import SinglePerimeter
from bikipy.perimeter.mixin import TrialWithPerimeterMixin


class PhysicalObjectTrialMixin(TrialWithPerimeterMixin, ABC):
    physical_object_inspect: bool = False

    animal_for_physical_object_profile: ClassVar[Literal["rodent"]] = ...
    physical_object_labels: ClassVar[set[str]] = set()
    all_perimeters_are_physical_objects: ClassVar[bool] = True

    @property
    @abstractmethod
    def all_physical_object_perimeters(self) -> tuple[SinglePerimeter, ...]:
        ...

    @classmethod
    @property
    def physical_object_set_class(cls) -> PhysicalObjectSetCLS:
        if cls.animal_for_physical_object_profile is ...:
            msg = "animal_for_physical_object_profile when working with PhysicalObjectTrialMixin Trials"
            raise ValueError(msg)

        return ANIMAL_LABEL_TO_PHYSICAL_OBJECT_SET_CLASS[cls.animal_for_physical_object_profile]

    @cached_property
    def perimeters(self):
        return self.all_physical_object_perimeters

    @property
    def _trial_feature_series_list(self) -> list[pd.Series]:
        upstream_list = super()._trial_feature_series_list
        upstream_list.append(self.physical_object_set.feature_summary)
        return upstream_list

    @cached_property
    def physical_object_keyword_arguments(self) -> dict[str, Any]:
        # TODO: Waiting for python 3.11 for variadic generic
        return {
            "reader": self.reader,
            "trial_obj_label": self.label,
            "inspect_arg": self.inspect_arg,
            "manual_video": self.video,
        }

    @cached_property
    def physical_object_set(self) -> PhysicalObjectSet:
        return self.physical_object_set_class.from_perimeter(
            *self.all_physical_object_perimeters, **self.physical_object_keyword_arguments
        )
