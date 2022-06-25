from functools import cached_property
from pathlib import Path

import pandas as pd
from compress_pickle import compress_pickle

from bikipy.core.base_class import BaseBikipy


class Analysis(BaseBikipy):
    combined_feature_motion_df: pd.DataFrame

    def save(self) -> None:
        save_root = self.inspect_directory or Path(".").resolve()
        compress_pickle.dump(self, save_root / f"analysis.pickle.lzma")

    @cached_property
    def trialwise_df(self) -> pd.DataFrame:
        return self.combined_feature_motion_df
