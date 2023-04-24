from logging import getLogger
from typing import Any, Iterable, Sequence

import numpy as np
from pydantic import validate_arguments
from pydantic_numpy import NDArray
from pydantic_numpy.dtype import NDArrayFp64

logger = getLogger(__name__)


def unique_with_counts_zipped(array: NDArray):
    return zip(*np.unique(array, return_counts=True))


@validate_arguments
def exclude_value_from_sequence(sequence: NDArrayFp64, exclude: Any):
    return sequence[sequence != exclude]


def feature_2d_multi_indexer(feature: str, groups: Iterable[str]):
    return [(str(feature), str(group)) for group in groups]


@validate_arguments
def reduce_repeating_sequences(
    repeating_sequence: NDArray,
    frame_tolerance: int,
    connector_element: Any = None,
) -> list:
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
    try:
        i = np.where(repeating_sequence != repeating_sequence[frame_tolerance - 1])[0][0]
    except IndexError:
        # The sequence consists only of one value after index "frame_tolerance - 1"
        assert len(repeating_sequence) > frame_tolerance - 1
        return [repeating_sequence[frame_tolerance - 1]]

    last_index = len(repeating_sequence) - frame_tolerance
    reduced_sequence = [(last_element := repeating_sequence[i])]
    while i + frame_tolerance < last_index:
        while True:
            i += 1
            new_element = repeating_sequence[i]
            if i + frame_tolerance == last_index or last_element != new_element:
                if np.mean(repeating_sequence[i : i + frame_tolerance] == new_element) > 0.5:
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


def blanket_experiment_label_generator(experiment_label: str) -> set[str]:
    return {
        experiment_label,
        f"{experiment_label}-enclosed",
        f"blanket-{experiment_label}",
        f"generic-{experiment_label}",
    }


def blanket_enclosed_experiment_label_generator(experiment_label: str) -> set[str]:
    upstream = blanket_experiment_label_generator(experiment_label)
    upstream.add(f"{experiment_label.capitalize()}Enclosed")
    return upstream
