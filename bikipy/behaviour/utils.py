from typing import AnyStr, Sequence, List
import itertools as it
from functools import lru_cache

import numpy as np


def unique_with_counts_zipped(array):
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


def reduce_str_sequence(str_sequence: Sequence) -> List:
    """
    Reduce consecutive sub-sequences in string sequence

    Parameters
    ----------
    str_sequence
        Sequence of strings

    Returns
    -------
    List, reduced string sequence; (A, A, A, B, B, C) -> [A, B, C]
    """

    reduced_str_sequence = [(current_str := str_sequence[0])]
    for string_element in str_sequence[1:]:
        if isinstance(string_element, str) and string_element != current_str:
            reduced_str_sequence.append((current_str := string_element))

    return reduced_str_sequence
