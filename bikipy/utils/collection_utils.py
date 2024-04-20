from itertools import chain
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic_numpy.typing import NpNDArray


def get_first(struct):
    return next(iter(struct))


def chain_lists_to_tuple(lists: Iterable[list]) -> tuple:
    return tuple(chain.from_iterable(lists))


def chain_iterables_to_multi_index(iterables: Iterable[Iterable[tuple[str, ...]]]) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(chain.from_iterable(iterables))


def max_len_in_iterable(iterable: Iterable[Sequence]):
    return max((len(feature_header) for feature_header in iterable))


def ndarray_to_tuple(array: NpNDArray):
    return tuple(map(tuple, array))


def evenly_spaced_indices_from_sequence(sequence: Sequence, number_of_elements: int):
    # https://stackoverflow.com/a/50685454/9793651
    return np.round(np.linspace(0, len(sequence) - 1, number_of_elements)).astype(int)


def evenly_spaced_indices(sequence_length: int, number_of_elements: int):
    # https://stackoverflow.com/a/50685454/9793651
    return np.round(np.linspace(0, sequence_length - 1, number_of_elements)).astype(int)


def project_mask_to_original(
    mask: NpNDArray, original: NpNDArray, original_mask: Optional[NpNDArray] = None
) -> NpNDArray:
    result = np.empty_like(original, dtype=mask.dtype)
    result[original_mask if original_mask is not None else ~original] = mask
    return result


def flatten_sequence(sequence: Sequence) -> NpNDArray:
    return np.asarray(sequence).reshape(-1)


def get_first_key_in_dict(source: dict) -> Any:
    return next(iter(source.keys()))


def get_first_value_in_dict(source: dict) -> Any:
    return next(iter(source.values()))


def dict_deep_update(source: dict, subsumed: dict) -> dict:
    """
    Update source dictionary, while updating dictionaries that are nested inside, recursively.
    The subsumed dictionary has priority.

    :param source: Source dictionary.
    :param subsumed: Dictionary that updates the source.
    :return: New dictionary.
    """
    for key, value in subsumed.items():
        if isinstance(value, dict):
            source[key] = dict_deep_update(source.get(key, {}), value)
        else:
            source[key] = value
    return source


def apply_slice_on_slice(source: slice, target: slice) -> slice:
    return slice(target.start - source.start, target.stop - source.stop)
