import itertools as it
from logging import getLogger
from typing import AnyStr, List, Sequence, SupportsInt

import numpy as np

logger = getLogger(__name__)


ARM_STRING_LABELS = (1, 2, 3)
CENTER_LABEL = 4


def unique_with_counts_zipped(array):
    return zip(*np.unique(array, return_counts=True))


def exclude_value_from_sequence(sequence: Sequence, exclude: AnyStr):
    sequence = np.asarray(sequence)
    return sequence[sequence != exclude]


def triplet_permutation_vs_base_permutation_dictionary(base_triplets: Sequence):
    """

    Parameters
    ----------
    base_triplets

    Returns
    -------

    """
    if isinstance(base_triplets, str):
        base_triplets = it.combinations_with_replacement(base_triplets, 3)
    return {
        base_triplet: set(it.permutations(base_triplet, 3))
        for base_triplet in base_triplets
    }


def reduce_repeating_sequences(
    str_sequence: Sequence, tolerance: SupportsInt = 6
) -> Sequence:
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
    if (unique_labels := len(labels)) <= 1:
        logger.info(f"Found on {unique_labels} unique labels, reducing to {labels}")
        return labels

    i, no_center_entry = 0, 0
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
            if current_str in ARM_STRING_LABELS and next_str in ARM_STRING_LABELS:
                reduced_str_sequence.append(CENTER_LABEL)
                no_center_entry += 1

            reduced_str_sequence.append(next_str)
            i += 1

    if no_center_entry:
        print(
            f"No center entry between entries from one arm to another, {no_center_entry}"
        )

    return reduced_str_sequence
