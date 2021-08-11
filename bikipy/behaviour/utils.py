import itertools as it
from collections.abc import Sequence
from logging import getLogger
from typing import Any, Iterable, Union, Optional

import numpy as np
from numba import njit

logger = getLogger(__name__)


def unique_with_counts_zipped(array):
    return zip(*np.unique(array, return_counts=True))


def exclude_value_from_sequence(sequence: Iterable, exclude: str):
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


@njit
def reduce_repeating_sequences(
    repeating_sequence: Sequence,
    frame_tolerance: Any,
    connector_element: Union[Any, None] = None,
) -> list[Optional[Any]]:
    """
    Reduce consecutive sub-sequences in string sequence

    Parameters
    ----------
    repeating_sequence
        Sequence that has repeating elements
    frame_tolerance
        frame_tolerance for changing current repeating element
    connector_element

    Returns
    -------
    list, reduced sequence; (A, A, A, B, B, C) -> [A, B, C]
    """
    # if (unique := np.unique(repeating_sequence)).size == 1:
    #     return unique

    i = 0
    last_index = len(repeating_sequence) - frame_tolerance
    reduced_sequence = []
    while i < last_index:
        current_element = repeating_sequence[i]
        # Skip element that is already first in repeating sequence
        while reduced_sequence and current_element == reduced_sequence[-1]:
            i += 1
            current_element = repeating_sequence[i]
            if i == last_index:
                return reduced_sequence

        while True:
            i += 1
            if i == last_index or current_element != repeating_sequence[i]:
                if (
                    np.mean(
                        repeating_sequence[i : i + frame_tolerance] == current_element
                    )
                    >= 0.6
                ):
                    if connector_element and reduced_sequence[-1] != connector_element:
                        reduced_sequence.append(connector_element)
                    reduced_sequence.append(current_element)
                    break
                if i == last_index:
                    break

    return reduced_sequence
