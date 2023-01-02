from abc import ABC
from functools import cached_property
from typing import Optional

from pydantic import BaseModel

from bikipy.core.typing import TrialId
from bikipy.utils.plot.inspect import InspectArg


class AbstractTrial(BaseModel, ABC):
    label: Optional[TrialId]
    inspect_arg: Optional[InspectArg]

    class Config:
        keep_untouched = (cached_property,)

    # @property
    # @abstractmethod
    # def reader(self) -> Reader:
    #     ...
    #
    # @property
    # @abstractmethod
    # def kinematic_coordinates(self) -> NDArrayFp64:
    #     ...
    #
    # @property
    # @abstractmethod
    # def video(self) -> VideoMetadata:
    #     ...
    #
    # @property
    # @abstractmethod
    # def _trial_feature_series_list(self) -> list[pd.Series]:
    #     ...
