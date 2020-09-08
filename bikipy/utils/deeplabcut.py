from typing import Sequence, AnyStr

from pandas.core.frame import DataFrame as DataFrameType
import numpy as np


def reduce_likelihoods(df: DataFrameType, regions_of_interest: Sequence) -> float:
    """
    Reduce likelihood values by multiplication; R^n to scalar

    :param df: DeepLabCut DataFrame with the likelihood values
    :param regions_of_interest: Regions of interest to reduce the likelihood of
    :return: Reduced likelihood values
    """
    return np.multiply.reduce(
        [df.loc[:, [(point, "likelihood")]].values for point in regions_of_interest]
    )


def get_region_of_interest_data(df: DataFrameType, region_of_interest: AnyStr):
    """
    Returns an np.ndarray with the coordinates of region of interest vs frames

    :param df: DeepLabCut DataFrame
    :param region_of_interest: Region of interest label to get coordinates of
    :return:
    """
    return df.loc[:, [(region_of_interest, "x"), (region_of_interest, "y")]].values
