from typing import ClassVar

from pydantic import BaseModel


class OpenFieldTrialMixin(BaseModel):
    trial_has_feature_frame: ClassVar[bool] = False
