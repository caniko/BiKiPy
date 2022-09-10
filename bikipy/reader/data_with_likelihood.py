import os
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Hashable, Iterable, Optional

import numpy as np
import pandas as pd
from pydantic import Field
from pydantic_numpy.dtype import NDArrayBool

from bikipy.reader.base import BaseReader
from bikipy.reader.utils import compute_midpoint_label

DEEPLABCUT_DF_INIT_KWARGS = {
    "index_col": 0,
    "skiprows": 1,
    "header": [0, 1],
    "na_filter": False,
    "dtype": {"coords": int, "x": float, "y": float, "likelihood": float},
}

CROPPING_PARAMETERS_BASE = {"x1": None, "x2": None, "y1": None, "y2": None}


logger = getLogger(__name__)


class DataWithLikelihoodReader(BaseReader):
    """
    Class that stores information about a given experiment conducted with DeepLabCut
    """

    min_likelihood: float = Field(
        0.80,
        description=(
            "The minimum likelihood the coordinates of the respective row. "
            "If below the values, the coords are discarded while being replaced "
            "by numpy.NaN"
        ),
    )

    _df_needs_to_be_cleaned = True

    def _isolate_coordinates(self, item):
        # remove likelihood column
        return np.delete(self.df[item].values, 2, 1)

    @cached_property
    def raw_df(self) -> pd.DataFrame:
        upstream_df = super().raw_df
        if self.df_path.suffix == ".h5":
            upstream_df = upstream_df.droplevel(0, axis=1)
        return upstream_df

    @cached_property
    def region_of_interest_to_boolean_index(self) -> dict[str, NDArrayBool]:
        return {roi: self.df[(roi, "likelihood")].values >= self.min_likelihood for roi in self.tracked_point_labels}

    @property
    def frames(self) -> int:
        return self.df.shape[0]

    def _compute_midpoint(
        self, df: pd.DataFrame, midpoint_group: Iterable[str], manual_midpoint_label: Optional[Hashable] = None
    ) -> pd.DataFrame:
        base = super()._compute_midpoint(df, midpoint_group, manual_midpoint_label)
        reduced_likelihoods = pd.Series(
            np.multiply.reduce(
                df.loc[:, pd.IndexSlice[midpoint_group, "likelihood"]].values,
                axis=1,
            ),
            index=df.index,
        )
        return pd.concat(
            (
                base,
                pd.DataFrame(
                    reduced_likelihoods,
                    columns=[(compute_midpoint_label(midpoint_group, manual_midpoint_label), "likelihood")],
                ),
            ),
            axis=1,
        )


class DeepLabCutReader(DataWithLikelihoodReader):
    # DeepLabCut datasets come with likelihoods, hence the alias; for user-friendliness
    pass


def convert_hdf_to_parquet(data_path, delete_hdf: bool = False):
    """
    Convert deeplabcut hdf files to parquet format, by replacing the filename suffix
    with parquet. Thereby, keeping the original path.

    :param data_path:
    :param delete_hdf:
    :return:
    """
    data_path = Path(data_path)
    parquet_path = data_path.with_suffix(".parquet")

    if not parquet_path.exists():
        pd.read_hdf(data_path, **DEEPLABCUT_DF_INIT_KWARGS).droplevel(0, axis=1).to_parquet(parquet_path)

    if delete_hdf:
        os.remove(data_path)

    return parquet_path
