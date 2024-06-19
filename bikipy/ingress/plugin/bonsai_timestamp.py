import numpy as np
import pandas as pd
from pydantic_numpy.typing import Np2DArrayFp64

from bikipy.core.typing import Label
from bikipy.ingress.plugin.core.base import BasePluginFile
from bikipy.ingress.plugin.core.mixins import TrialWiseMetadataOnlyMixin


class PluginBonsaiTimestamp(TrialWiseMetadataOnlyMixin, BasePluginFile):
    ingress_key = "timestamp"
    code_key = "timestamp"
    default_trial_argument_key = "coordinate_timestamp_index"
    human_readable_index = "Timestamp"

    def trialwise_and_metadata(self, trial_id: Label, naive: bool = False) -> Np2DArrayFp64:
        self._assert_correct_scope_trialwise_metadata()
        datetime_series = pd.read_csv(self.data_path, header=None, usecols=[16], parse_dates=[0]).iloc[:, 0]
        datetime_array = datetime_series.values
        return (datetime_array - datetime_array[0]).astype(np.float64) / 10**6
