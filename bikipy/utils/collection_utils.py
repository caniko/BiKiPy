from functools import lru_cache
from itertools import chain
from typing import Any, Iterable, Optional, Sequence, Callable, List, Tuple

import numpy as np
import pandas as pd
from pydantic_numpy.dtype import NDArray


def chain_lists_to_tuple(lists: Iterable[list]) -> tuple:
    return tuple(chain(*lists))


def chain_iterables_to_multi_index(iterables: Iterable[Iterable[tuple[str, ...]]]) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(chain(*iterables))


def max_len_in_iterable(iterable: Iterable[Sequence]):
    return max((len(feature_header) for feature_header in iterable))


def ndarray_to_tuple(array: NDArray):
    return tuple(map(tuple, array))


def add_filler_to_sequence(
    sequence: Iterable[Sequence[str]],
    filler: str | Iterable[str],
    on_start: bool = True,
) -> list[tuple[str, ...]]:
    if isinstance(filler, str):
        filler = [filler]
    return [(*filler, *headers) if on_start else (*headers, *filler) for headers in sequence]


def add_n_levels_to_multi_index(
    multi_index: pd.Index | pd.MultiIndex,
    n_levels: int,
    filler: Optional[str] = None,
    on_start: bool = True,
) -> pd.MultiIndex:
    multi_index_as_tuples = list(multi_index)
    levels_to_add = [filler or "" for _ in range(n_levels)]

    if isinstance(multi_index, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(add_filler_to_sequence(multi_index_as_tuples, levels_to_add))

    elif isinstance(multi_index, pd.Index):
        return pd.MultiIndex.from_tuples(
            [(*levels_to_add, headers) if on_start else (headers, *levels_to_add) for headers in multi_index_as_tuples]
        )


def copycat_assumes_levels_of_icon(copycat: pd.DataFrame, icon: pd.DataFrame, filler: Optional[str] = None):
    assert copycat.columns.nlevels < icon.columns.nlevels
    clone_df = copycat.copy()
    clone_df.columns = add_n_levels_to_multi_index(
        clone_df.columns, icon.columns.nlevels - copycat.columns.nlevels, filler=filler
    )
    clone_df.columns.names = icon.columns.names
    return clone_df


@lru_cache
def generic_multi_indexer(*basis_labels) -> Callable[[Any, int], list[tuple[Any, Any, Any]]]:
    number_of_levels = 1 if isinstance(basis_labels[0], str) else len(basis_labels[0])
    assert not any(number_of_levels != 1 if isinstance(label, str) else len(label) for label in basis_labels)

    def result(category: Any, desired_nlevel: int):
        # if desired_nlevel < number_of_levels:
        if (number_of_levels_to_add := desired_nlevel - number_of_levels - 1) < 0:
            msg = f"Desired number of levels, {desired_nlevel}, is lower than the initial, {number_of_levels}"
            raise ValueError(msg)
        levels_to_add = ["" for _ in range(number_of_levels_to_add)]
        return [(category, label, *levels_to_add) for label in basis_labels]

    return result


@lru_cache
def flatten_multi_index(indices: Iterable[Sequence[str]]) -> tuple[str, ...]:
    return tuple("-".join(index) for index in indices)


def evenly_spaced_indices_from_sequence(sequence: Sequence, number_of_elements: int):
    # https://stackoverflow.com/a/50685454/9793651
    return np.round(np.linspace(0, len(sequence) - 1, number_of_elements)).astype(int)


def evenly_spaced_indices(sequence_length: int, number_of_elements: int):
    # https://stackoverflow.com/a/50685454/9793651
    return np.round(np.linspace(0, sequence_length - 1, number_of_elements)).astype(int)


def project_mask_to_original(mask: NDArray, original: NDArray, original_mask: Optional[NDArray] = None) -> NDArray:
    result = np.empty_like(original, dtype=mask.dtype)
    result[original_mask if original_mask is not None else ~original] = mask
    return result


def flatten_sequence(sequence: Sequence) -> NDArray:
    return np.asarray(sequence).reshape(-1)


def get_first_key_in_dict(source: dict) -> Any:
    return next(iter(source.keys()))


def get_first_value_in_dict(source: dict) -> Any:
    return next(iter(source.values()))
