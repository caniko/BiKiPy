import collections.abc as abc
import os
from functools import cached_property
from logging import getLogger
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pydantic import Field

from bikipy.feature.midpoint import recursive_midpoint
from bikipy.reader.base import BaseReader

DEEPLABCUT_DF_INIT_KWARGS = {
    "index_col": 0,
    "skiprows": 1,
    "header": [0, 1],
    "na_filter": False,
    "dtype": {"coords": int, "x": float, "y": float, "likelihood": float},
}

CROPPING_PARAMETERS_BASE = {"x1": None, "x2": None, "y1": None, "y2": None}


logger = getLogger(__name__)


class DeepLabCutReader(BaseReader):
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
    def raw_df(self):
        if not self._df_needs_to_be_cleaned:
            return super().raw_df

        if self.df_path.suffix == ".csv":
            return pd.read_csv(self.df_path, **DEEPLABCUT_DF_INIT_KWARGS)
        elif self.df_path.suffix == ".h5":
            return pd.read_hdf(self.df_path, **DEEPLABCUT_DF_INIT_KWARGS).droplevel(
                0, axis=1
            )
        else:
            # DeepLabCut doesn't support other formats natively
            logger.debug(
                f"{self.df_path.suffix}, is not natively supported by DeepLabCut, "
                f"assuming user has manually cleaned and exported the data file"
                f"to another format that is supported by BiKiPy.BaseReader. Good luck"
            )
            return super().raw_df

    @cached_property
    def augmented(self):
        result = self.raw_df.copy()
        if self.x_axis_crop_end_point:
            for roi in self.tracked_point_labels:
                result.loc[:, (roi, "x")] = (
                    result.loc[:, (roi, "x")] + self.x_axis_crop_end_point
                )

        if self.y_add:
            for roi in self.tracked_point_labels:
                result.loc[:, (roi, "y")] = result.loc[:, (roi, "y")] + self.y_add

        if self.midpoint_groups:
            midpoint_data, midpoint_based_midpoints = {}, {}
            for name, group in self.midpoint_groups.items():
                if all(component in self.tracked_point_labels for component in group):
                    group_points = [
                        self.get_tracking_data(component_name)
                        for component_name in group
                    ]
                    midpoint_x, midpoint_y = recursive_midpoint(group_points).T
                    midpoint_data[(name, "x")] = midpoint_x
                    midpoint_data[(name, "y")] = midpoint_y
                    midpoint_data[(name, "likelihood")] = self.reduce_likelihoods(group)
                elif all(
                    component in self.tracked_and_midpoint_labels for component in group
                ):
                    midpoint_based_midpoints[name] = group
                else:
                    msg = (
                        f"Midpoint {name}, cannot be derived as its components are "
                        f"not defined in the tracked dataset nor in midpoint_groups"
                    )
                    raise ValueError(msg)

            for name, group in midpoint_based_midpoints.items():
                new_midpoint_likelihood = None
                group_points = []
                for component_name in group:
                    if component_name in self.midpoint_groups:
                        component_likelihood = midpoint_data[
                            (component_name, "likelihood")
                        ]
                        group_points.append(
                            np.array(
                                (
                                    midpoint_data[(component_name, "x")],
                                    midpoint_data[(component_name, "y")],
                                )
                            ).T
                        )
                    else:
                        group_points.append(self.get_tracking_data(component_name))
                        component_likelihood = self.raw_df.loc[
                            :, [(component_name, "likelihood")]
                        ].values.T[0]

                    if new_midpoint_likelihood is not None:
                        new_midpoint_likelihood *= component_likelihood
                    else:
                        new_midpoint_likelihood = component_likelihood

                midpoint_x, midpoint_y = recursive_midpoint(group_points).T
                midpoint_data[(name, "x")] = midpoint_x
                midpoint_data[(name, "y")] = midpoint_y
                midpoint_data[(name, "likelihood")] = new_midpoint_likelihood.T[0]

            midpoint_df = pd.DataFrame.from_dict(midpoint_data)
            result = pd.concat((self.raw_df, midpoint_df), axis=1)

        return result

    @property
    def tracked_point_labels(self) -> tuple:
        return tuple(self.raw_df.columns.levels[0])

    @cached_property
    def tracked_and_midpoint_labels(self):
        return *self.tracked_point_labels, *self.midpoint_groups.keys()

    @cached_property
    def region_of_interest_vs_boolean_index(self):
        return {
            roi: self.df[(roi, "likelihood")].values >= self.min_likelihood
            for roi in self.tracked_point_labels
        }

    @property
    def frames(self):
        return self.df.shape[0]

    def reduce_likelihoods(self, tracked_point_labels: Sequence) -> np.ndarray:
        """
        Reduce likelihood values by multiplication; R^n to scalar

        :param tracked_point_labels: Regions of interest of which will have its
        likelihood values reduced
        :return: np.ndarray with the reduced likelihood values
        """
        return np.multiply.reduce(
            [
                self.raw_df.loc[:, [(point, "likelihood")]].values
                for point in tracked_point_labels
            ]
        ).T[0]

    def get_tracking_data(self, label: str):
        """Returns an np.ndarray with the coordinates of label"""
        return self.raw_df.loc[:, [(label, "x"), (label, "y")]].values


def convert_hdf_to_parquet(data_paths, delete_hdf: bool = False):
    """
    Convert deeplabcut hdf files to parquet format, by replacing the filename suffix
    with parquet. Thereby, keeping the original path.

    :param data_paths:
    :param delete_hdf:
    :return:
    """
    data_path = Path(data_paths)
    parquet_path = data_path.with_suffix(".parquet")

    if not parquet_path.exists():
        pd.read_hdf(data_path, **DEEPLABCUT_DF_INIT_KWARGS).droplevel(
            0, axis=1
        ).to_parquet(parquet_path)

    if delete_hdf:
        os.remove(data_path)

    return parquet_path
