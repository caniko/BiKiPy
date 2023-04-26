import numpy as np
import pandas as pd
from pydantic_numpy import NDArray

from bikipy.core.typing import Label
from bikipy.ingress.plugin.base import BasePluginFile, TrialWiseMetadataOnlyMixin


class PluginBonsaiTimestamp(TrialWiseMetadataOnlyMixin, BasePluginFile):
    ingress_key = "timestamp"
    code_key = "timestamp"
    default_trial_argument_key = "coordinate_timestamp_index"
    human_readable_index = "Timestamp"

    def trialwise_and_metadata(self, trial_id: Label, naive: bool = False) -> NDArray:
        self._assert_correct_scope_trialwise_metadata()
        datetime_array = (
            pd.read_csv(self.data_path, header=None, usecols=[16], parse_dates=[0]).values.T[0].astype(np.datetime64)
        )
        return (datetime_array - datetime_array[0]).astype(float) / 10**6
