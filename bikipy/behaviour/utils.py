import itertools as it
from collections.abc import Sequence
from logging import getLogger
from typing import Any, Iterable, Optional, Union

import numpy as np

logger = getLogger(__name__)


def unique_with_counts_zipped(array):
    return zip(*np.unique(array, return_counts=True))


def exclude_value_from_sequence(sequence: Iterable, exclude: Any):
    sequence = np.asarray(sequence)
    return sequence[sequence != exclude]


def feature_2d_multi_indexer(feature: str, groups: Iterable[str]):
    return [(feature, group) for group in groups]


def reduce_repeating_sequences(
    repeating_sequence: np.ndarray,
    frame_tolerance: Any,
    connector_element: Any = None,
) -> list[Any, ...]:
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
