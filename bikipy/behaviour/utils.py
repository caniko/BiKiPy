import itertools as it
from collections.abc import Sequence
from logging import getLogger
from typing import Any, Iterable, Optional, Union

import numpy as np


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

    repeating_sequence = np.asarray(repeating_sequence)

    try:
        i = np.where(repeating_sequence != repeating_sequence[frame_tolerance - 1])[0][
            0
        ]
    except IndexError:
        # The sequence consists only of one value after index "frame_tolerance - 1"
        assert len(repeating_sequence) > frame_tolerance - 1
        return [repeating_sequence[frame_tolerance - 1]]

    last_index = len(repeating_sequence) - frame_tolerance
    reduced_sequence = [(last_element := repeating_sequence[i])]
    while i + frame_tolerance < last_index:
        while True:
            i += 1
            if i + frame_tolerance == last_index or last_element != (
                new_element := repeating_sequence[i]
            ):
                if (
                    np.mean(repeating_sequence[i : i + frame_tolerance] == new_element)
                    >= 0.6
                ):
                    if connector_element and reduced_sequence[-1] != connector_element:
                        reduced_sequence.append(connector_element)
                    reduced_sequence.append(new_element)
                    last_element = new_element
                    break
                if i + frame_tolerance == last_index:
                    break

    return reduced_sequence


def reduce_repeating_sequences_absolute(
    repeating_sequence: Sequence,
    frame_tolerance: Any,
):
    last_index = len(repeating_sequence) - frame_tolerance
    reduced_sequence = [(last_element := repeating_sequence[0])]
    for i in range(1, last_index):
        if not any(
            tolerated_element == last_element
            for tolerated_element in repeating_sequence[i - frame_tolerance + 1 : i + 1]
        ):
            reduced_sequence.append(last_element)

    return reduced_sequence
