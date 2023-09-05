import os
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Generic, Hashable, Iterable, Optional

import numpy as np
import pandas as pd
from pydantic import Field, FilePath, computed_field
from pydantic_numpy.typing import NpNDArrayBool, NpNDArrayFp64

from bikipy.reader.base import BaseReader, Enclosure
from bikipy.reader.utils import compute_midpoint_label
from bikipy.utils.constants import TO_PARQUET_KWARGS

DEEPLABCUT_DF_INIT_KWARGS = {
    "index_col": 0,
    "skiprows": 1,
    "header": [0, 1],
    "na_filter": False,
    "dtype": {"coords": int, "x": float, "y": float, "likelihood": float},
}

CROPPING_PARAMETERS_BASE = {"x1": None, "x2": None, "y1": None, "y2": None}


logger = getLogger(__name__)


class DataWithLikelihoodReader(BaseReader[Enclosure], Generic[Enclosure]):
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

    @staticmethod
    def isolate_coordinates_from_native_df(df: pd.DataFrame, key: Iterable[str] | str) -> NpNDArrayFp64:
        # remove likelihood column
        return np.delete(df[key].values, 2, 1)

    @computed_field  # type: ignore[misc]
    @cached_property
    def region_of_interest_to_boolean_index(self) -> dict[str, NpNDArrayBool]:
        return {roi: self.df[(roi, "likelihood")].values >= self.min_likelihood for roi in self.all_tracked_labels}

    @computed_field  # type: ignore[misc]
    @property
    def frames(self) -> int:
        return self.df.shape[0]


class DeepLabCutReader(DataWithLikelihoodReader[Enclosure], Generic[Enclosure]):
    def _read_hdf(self, path: FilePath) -> pd.DataFrame:
        df = pd.read_hdf(path)
        df.columns = df.columns.droplevel()
        return df


def convert_hdf_to_parquet(data_path, delete_hdf: bool = False, ignore_pre_existing: bool = False) -> Path:
    """
    Convert deeplabcut hdf files to parquet format, by replacing the filename suffix
    with parquet. Thereby, keeping the original path.

    :param data_path:
    :param delete_hdf:
    :return:
    """
    data_path = Path(data_path)
    parquet_path = data_path.with_suffix(".parquet")

    df = pd.read_hdf(data_path, **DEEPLABCUT_DF_INIT_KWARGS).droplevel(0, axis=1)

    if ignore_pre_existing or not parquet_path.exists():
        df.to_parquet(parquet_path, **TO_PARQUET_KWARGS)

    if delete_hdf:
        os.remove(data_path)

    return parquet_path
