from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, ClassVar

import pandas as pd

from bikipy.feature.physical_object.set import PhysicalObjectSetCLS, PhysicalObjectSet, GenericPhysicalObjectSet
from bikipy.perimeter.base import SinglePerimeter
from bikipy.perimeter.mixin import TrialWithPerimeterMixin


class PhysicalObjectTrialMixin(TrialWithPerimeterMixin, ABC):
    physical_object_inspect: bool = False

    all_perimeters_are_physical_objects: ClassVar[bool] = True

    @property
    @abstractmethod
    def physical_object_perimeters(self) -> tuple[SinglePerimeter, ...]:
        ...

    @classmethod
    @property
    def physical_object_set_class(cls) -> PhysicalObjectSetCLS:
        match cls.animal_profile:
            case "rodent":
                from bikipy.feature.physical_object.single.observation_qualia.rodent import RodentObservationQualia
                from bikipy.behaviour.core import Trial

                GenericPhysicalObjectSet.update_forward_refs(Trial=Trial)

                class RodentPhysicalObjectSet(GenericPhysicalObjectSet[RodentObservationQualia]):
                    physical_object_profile_class = RodentObservationQualia

                return RodentPhysicalObjectSet

    @cached_property
    def perimeters(self):
        return self.physical_object_perimeters

    @property
    def _trial_feature_series_list(self) -> list[pd.Series]:
        upstream_list = super()._trial_feature_series_list
        upstream_list.append(self.physical_object_set.feature_summary)
        return upstream_list

    @cached_property
    def physical_object_component_kwargs(self) -> dict[str, Any]:
        # TODO: Waiting for python 3.11 for variadic generic
        return {
            "reader": self.reader,
            "trial_obj_label": self.label,
            "inspect_arg": self.inspect_arg,
        }

    @cached_property
    def physical_object_set(self) -> PhysicalObjectSet:
        return self.physical_object_set_class(trial=self, manual_video=self.video)
