from itertools import chain
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import numpy.typing as nt
import pandas as pd
from pydantic_numpy.dtype import NDArray


def get_first(struct):
    return next(iter(struct))


def chain_lists_to_tuple(lists: Iterable[list]) -> tuple:
    return tuple(chain(*lists))


def chain_iterables_to_multi_index(iterables: Iterable[Iterable[tuple[str, ...]]]) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(chain(*iterables))


def max_len_in_iterable(iterable: Iterable[Sequence]):
    return max((len(feature_header) for feature_header in iterable))


def ndarray_to_tuple(array: NDArray):
    return tuple(map(tuple, array))


def evenly_spaced_indices_from_sequence(sequence: Sequence, number_of_elements: int):
    # https://stackoverflow.com/a/50685454/9793651
    return np.round(np.linspace(0, len(sequence) - 1, number_of_elements)).astype(int)


def evenly_spaced_indices(sequence_length: int, number_of_elements: int):
    # https://stackoverflow.com/a/50685454/9793651
    return np.round(np.linspace(0, sequence_length - 1, number_of_elements)).astype(int)


def project_mask_to_original(mask: NDArray, original: NDArray, original_mask: Optional[NDArray] = None) -> nt.NDArray:
    result = np.empty_like(original, dtype=mask.dtype)
    result[original_mask if original_mask is not None else ~original] = mask
    return result


def flatten_sequence(sequence: Sequence) -> nt.NDArray:
    return np.asarray(sequence).reshape(-1)


def get_first_key_in_dict(source: dict) -> Any:
    return next(iter(source.keys()))


def get_first_value_in_dict(source: dict) -> Any:
    return next(iter(source.values()))
