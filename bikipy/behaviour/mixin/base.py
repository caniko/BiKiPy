from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel


class FeaturefullTrialMixin(BaseModel, ABC):
    trial_has_feature_frame: ClassVar[bool] = True
    feature_summary_column: ClassVar[list] = []

    @property
    @abstractmethod
    def feature_summary_row(self) -> list:
        ...
