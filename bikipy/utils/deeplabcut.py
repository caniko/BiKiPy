from collections.abc import Sequence

import numpy as np
from pandas.core.frame import DataFrame as DataFrameType


def reduce_likelihoods(df: DataFrameType, regions_of_interest: Sequence) -> np.ndarray:
    """
    Reduce likelihood values by multiplication; R^n to scalar

    Parameters
    ----------
    df
        DeepLabCut DataFrame
    regions_of_interest
        Regions of interest of which will have its likelihood values reduced

    Returns
    -------
    np.ndarray with the reduced likelihood values
    """
    return np.multiply.reduce(
        [df.loc[:, [(point, "likelihood")]].values for point in regions_of_interest]
    )


def get_region_of_interest_data(df: DataFrameType, region_of_interest: str):
    """
    Returns an np.ndarray with the coordinates of region of interest vs frames

    Parameters
    ----------
    df
        DeepLabCut DataFrame
    region_of_interest
        Region of interest label to get coordinates of

    Returns
    -------

    """
    return df.loc[:, [(region_of_interest, "x"), (region_of_interest, "y")]].values
