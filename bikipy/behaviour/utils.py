import itertools as it
from functools import lru_cache
from typing import AnyStr, List, Sequence, SupportsInt

import numpy as np

ARM_STRING_LABELS = ("A", "B", "C")


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


def reduce_str_sequence(str_sequence: Sequence, tolerance: SupportsInt = 6) -> List:
    """
    Reduce consecutive sub-sequences in string sequence

    Parameters
    ----------
    str_sequence
        Sequence of strings
    tolerance

    Returns
    -------
    List, reduced string sequence; (A, A, A, B, B, C) -> [A, B, C]
    """

    tolerance = int(tolerance)

    labels = np.unique(str_sequence)
    assert 2 <= len(labels) <= 4

    center_label = None
    for label in labels:
        if label not in ARM_STRING_LABELS:
            center_label = label
            break
    assert center_label

    i = 0
    max_len = len(str_sequence) - tolerance
    reduced_str_sequence = []
    while i < max_len:
        current_str = str_sequence[i]

        tolerable = True
        for following_idx in range(i + 1, i + 1 + tolerance):
            i = following_idx
            if str_sequence[following_idx] != current_str:
                tolerable = False
                break
        if tolerable:
            reduced_str_sequence.append(str_sequence[i])
            break

    assert reduced_str_sequence

    i += 1
    while i < max_len:
        if (current_str := reduced_str_sequence[-1]) == (next_str := str_sequence[i]):
            i += 1
            continue

        tolerable = True
        for following_idx in range(i + 1, i + 1 + tolerance):
            if str_sequence[following_idx] == current_str:
                tolerable = False
                i = following_idx
                break
        if tolerable:
            if (
                current_str.upper() in ARM_STRING_LABELS
                and next_str.upper() in ARM_STRING_LABELS
            ):
                reduced_str_sequence.append(center_label)

            reduced_str_sequence.append(next_str)
            i += 1

    return reduced_str_sequence
