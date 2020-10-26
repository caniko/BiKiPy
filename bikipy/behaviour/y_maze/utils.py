from typing import AnyStr, Sequence, List
import itertools as it
from functools import lru_cache

import numpy as np


def unique_with_counts_zipped(array):
    array = np.asanyarray(array)
    return zip(*np.unique(array, return_counts=True))


def exclude_value_from_sequence(sequence: Sequence, exclude: AnyStr):
    sequence = np.asanyarray(sequence)
    return sequence[sequence != exclude]


@lru_cache
def triplet_permutation_vs_base_permutation_dictionary(base_triplets: Sequence):
    """

    Parameters
    ----------
    base_triplets

    Returns
    -------

    """
    result = {}
    if isinstance(base_triplets, str):
        base_triplets = it.combinations_with_replacement(base_triplets, 3)
    for base_triplet in base_triplets:
        base_triplet = "".join(base_triplet)
        for permutation in set(it.permutations(base_triplet, 3)):
            result["".join(permutation)] = base_triplet
    return result


def reduce_location_sequence(location_per_frame: Sequence) -> List:
    """
    Reduce the location per frame to a location sequence

    Parameters
    ----------
    location_per_frame
        Sequence of location on the respective frame

    Returns
    -------
    List containing the location sequence; (A, A, A, B, B, C) -> [A, B, C]
    """

    current_char = None
    arm_location_sequence = []
    for location in location_per_frame:
        if isinstance(location, str) and (location != current_char or not current_char):
            arm_location_sequence.append((current_char := location))

    return arm_location_sequence
