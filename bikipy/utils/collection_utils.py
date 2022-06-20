from itertools import chain
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
from pydantic import validate_arguments

from bikipy.core.typing import NDArray


def chain_lists_to_tuple(lists: Iterable[list]) -> tuple:
    return tuple(chain(*lists))


def max_len_in_iterable(iterable: Iterable[Sequence]):
    return max((len(feature_header) for feature_header in iterable))


def ndarray_to_tuple(array: NDArray):
    return tuple(map(tuple, array))


def add_n_levels_to_multi_index(
    multi_index: pd.Index | pd.MultiIndex, n_levels: int, on_start: bool = True, filler: Optional[str] = None
) -> pd.MultiIndex:
    multi_index_as_tuples = list(multi_index)
    levels_to_add = [filler or "" for _ in range(n_levels)]

    if isinstance(multi_index, pd.MultiIndex):
        if on_start:
            return pd.MultiIndex.from_tuples([(*levels_to_add, *headers) for headers in multi_index_as_tuples])
        return pd.MultiIndex.from_tuples([(*headers, *levels_to_add) for headers in multi_index_as_tuples])

    elif isinstance(multi_index, pd.Index):
        if on_start:
            return pd.MultiIndex.from_tuples([(*levels_to_add, headers) for headers in multi_index_as_tuples])
        return pd.MultiIndex.from_tuples([(headers, *levels_to_add) for headers in multi_index_as_tuples])


def copycat_assumes_levels_of_icon(copycat: pd.DataFrame, icon: pd.DataFrame, filler: Optional[str]):
    assert copycat.columns.nlevels < icon.columns.nlevels
    clone_df = copycat.copy()
    clone_df.columns = add_n_levels_to_multi_index(
        clone_df.columns, icon.columns.nlevels - copycat.columns.nlevels, filler
    )
    clone_df.columns.names = icon.columns.names
    return clone_df


def generic_multi_indexer(*basis_labels):
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
