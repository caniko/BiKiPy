from typing import Union, Sequence, List
import itertools

import numpy as np


def feature_scale(
    data: Sequence,
    real_min: Union[float, int, None] = None,
    real_max: Union[float, int, None] = None,
):
    """
    Scale the data to [0, 1]; 0 is the smallest and 1 is the highest

    Will find the minimum and maximum from data if no real values are provided

    :param data: Data sequence
    :param real_min: Manual definition of the minimum
    :param real_max: Manual definition of the maximum
    :return:
    """
    data = np.asanyarray(data)

    minimum = real_min or data.min()
    maximum = real_max or data.max()

    return (data - minimum) / maximum - minimum


def permutations_with_replacement(sequence: Sequence) -> List[str]:
    """
    Implementation of permutation with replacement

    Length  of result should be n^r:
        P^R(n,r) = n^r   For, n >= 0, and r >= 0

    Parameters
    ----------
    sequence : object
        Sequence that will have its permutation with replacement computed

    Returns
    -------
    Set, storing the permutation with replacement of sequence
    """
    # result = []
    # for comb in itertools.combinations_with_replacement(sequence, len(sequence)):
    #     result.extend(itertools.permutations(comb))
    # return set(result)

    return ["".join(x) for x in itertools.product(sequence, len(sequence))]


def invalidate_array(data, boolean_array):
    data[boolean_array] = np.nan
    return data
