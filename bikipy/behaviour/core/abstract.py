from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from pydantic import BaseModel
from pydantic_numpy import NDArrayFp64

from bikipy.core.typing import TrialId
from bikipy.core.video import VideoMetadata
from bikipy.reader.base import Reader
from bikipy.utils.plotting import InspectArg


class AbstractTrial(BaseModel, ABC):
    label: Optional[TrialId]
    inspect_arg: Optional[InspectArg]

    @abstractmethod
    @property
    def reader(self) -> Reader:
        ...

    @abstractmethod
    @property
    def kinematic_coordinates(self) -> NDArrayFp64:
        ...

    @abstractmethod
    @property
    def video(self) -> VideoMetadata:
        ...

    @abstractmethod
    @property
    def _trial_feature_series_list(self) -> list[pd.Series]:
        ...
