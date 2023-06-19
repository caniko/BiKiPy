from functools import cached_property, lru_cache
from typing import ClassVar, Optional

import numpy as np
from pydantic import validate_arguments, validator
from pydantic_numpy import NDArrayInt16
from skg import ngauss_fit

from bikipy._dev_utils.fields import enclosure_field
from bikipy._dev_utils.message import report_to_github
from bikipy.behaviour.core.base import BaseExperiment, BaseTrial, HabituationTrialMixin
from bikipy.perimeter.base import Perimeter, PerimeterCLS, PerimeterSet
from bikipy.reader.base import ReaderCLS


class EnclosedTrial(BaseTrial):
    manual_enclosure: Optional[Perimeter] = enclosure_field

    trial_perimeter_enclosure_class: ClassVar[PerimeterCLS]
    enclosure_perimeter_object_attribute_names: ClassVar[set[str]] = set()

    gaussian_dividend_multiplayer: ClassVar[int] = 1

    @validator("manual_enclosure")
    def manual_enclosure_is_instance_of_trial_perimeter_enclosure_class(cls, value: Perimeter):
        """
        This could be enforced through GenericModel, but GenericModel types are reserved for inter-trial perimeters,
        and not trial enclosures
        """
        if not isinstance(value, cls.trial_perimeter_enclosure_class):
            msg = f"manual_enclosure must be an instance of {cls.trial_perimeter_enclosure_class.__name__}"
            raise AttributeError(msg)
        return value

    @classmethod
    @property
    def reader_class(cls) -> ReaderCLS:
        return super().reader_class[cls.trial_perimeter_enclosure_class]

    @property
    def _reader_kwargs(self) -> dict:
        return {**super()._reader_kwargs, "trial_enclosure": self.enclosure}

    @cached_property
    def enclosure(self) -> Perimeter:
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
            msg = "No enclosure defined for EnclosureTrial"
            AttributeError(msg)

        if len(enclosures) == 1:
            return enclosures[0]

        return PerimeterSet(perimeters=enclosures)

    @cached_property
    def gaussian_center_to_periphery_score(self) -> float:
        func = gaussian_scoring_field(
            self.video.metric_resolution, gaussian_dividend_multiplayer=self.gaussian_dividend_multiplayer
        )
        scores = np.array(
            [func(*coordinate) for coordinate in self.reader.kinematic_coordinates if not np.any(np.isnan(coordinate))]
        )
        return np.sum(scores) / (self.gaussian_dividend_multiplayer * self.number_of_frames)


class EnclosedHabituationTrial(HabituationTrialMixin, EnclosedTrial):
    pass


class EnclosedExperiment(BaseExperiment):
    @classmethod
    @property
    def trial_perimeter_enclosure_classes(cls) -> dict[str, PerimeterCLS]:
        return {
            enclosed_trial_class.experiment_stage: enclosed_trial_class.trial_perimeter_enclosure_class
            for enclosed_trial_class in cls.trial_classes
            if issubclass(enclosed_trial_class, EnclosedTrial)
        }


@validate_arguments
@lru_cache
def gaussian_scoring_field(resolution: NDArrayInt16, scale: int = 1, gaussian_dividend_multiplayer: int = 1):
    resolution *= scale

    model = ngauss_fit.model(
        x=np.indices(resolution, dtype=float),
        a=gaussian_dividend_multiplayer,
        mu=resolution / 2.0,
        sigma=np.array([[resolution[0] ** 2, 0.0], [0.0, resolution[1] ** 2]]),
        axis=0,
    )

    scale_as_float = float(scale)
    return lambda x, y: model[round(x * scale_as_float)][round(y * scale_as_float)]
