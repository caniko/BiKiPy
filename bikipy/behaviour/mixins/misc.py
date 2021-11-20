from pydantic import BaseModel


class OpenFieldTrialMixin(BaseModel):
    _trial_has_feature_frame = False
