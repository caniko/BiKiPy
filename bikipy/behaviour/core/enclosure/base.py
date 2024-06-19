from functools import cached_property
from typing import ClassVar, Optional, Type

from pydantic import computed_field, field_validator

from bikipy._dev_utils.fields import enclosure_field
from bikipy._dev_utils.message import report_to_github
from bikipy.behaviour.core.base import BaseExperiment, BaseTrial, HabituationTrialMixin
from bikipy.behaviour.core.constant import ExperimentStage
from bikipy.perimeter.base import BasePerimeter, PerimeterCLS, PerimeterSet


class EnclosedTrial(BaseTrial):
    manual_enclosure: Optional[BasePerimeter] = enclosure_field

    trial_perimeter_enclosure_class: ClassVar[PerimeterCLS]
    enclosure_perimeter_object_attribute_names: ClassVar[set[str]] = set()

    gaussian_dividend_multiplayer: ClassVar[int] = 1

    @field_validator("manual_enclosure")
    def manual_enclosure_is_instance_of_trial_perimeter_enclosure_class(cls, value: BasePerimeter):
        """
        This could be enforced through but GenericModel types are reserved for inter-trial perimeters,
        and not trial enclosures
        """
        if not isinstance(value, cls.trial_perimeter_enclosure_class):
            msg = f"manual_enclosure must be an instance of {cls.trial_perimeter_enclosure_class.__name__}"
            raise AttributeError(msg)
        return value

    @computed_field  # type: ignore[misc]
    @property
    def _reader_kwargs(self) -> dict:
        return {**super()._reader_kwargs, "trial_enclosure": self.enclosure}

    @computed_field  # type: ignore[misc]
    @cached_property
    def enclosure(self) -> BasePerimeter | None:
        """
        Will be defined as trial_enclosure in `reader` when passed onto reader from self._reader_kwargs

        :return:
        """
        enclosures = []
        for name in self.enclosure_perimeter_object_attribute_names:
            try:
                enclosures.append(self.__getattribute__(name))
            except AttributeError as e:
                msg = (
                    f"{name} defined in enclosure_perimeter_object_attribute_names is not an attribute. "
                    f"{report_to_github}"
                )
                raise AttributeError(msg) from e
        if self.manual_enclosure:
            enclosures.append(self.manual_enclosure)

        if not enclosures:
            return

        if len(enclosures) == 1:
            return enclosures.pop()

        return PerimeterSet(perimeters=enclosures)


class EnclosedHabituationTrial(HabituationTrialMixin, EnclosedTrial):
    pass


class EnclosedExperiment(BaseExperiment):
    @classmethod
    @property
    def trial_perimeter_enclosure_classes(cls) -> dict[ExperimentStage, Type[BasePerimeter]]:
        return {
            enclosed_trial_class.experiment_stage: enclosed_trial_class.trial_perimeter_enclosure_class
            for enclosed_trial_class in cls.trial_classes
            if issubclass(enclosed_trial_class, EnclosedTrial)
        }
