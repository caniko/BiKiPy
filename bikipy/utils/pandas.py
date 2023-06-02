from functools import lru_cache
from typing import Any, Callable, Iterable, Optional, Sequence

import pandas as pd


def motion_analysis_indexer_for_subsection(category: Any, level: int):
    return generic_multi_indexer(
        "Displacement", "MedianSpeed", "MedianAcceleration", "FreezingTime", "Entries", "SecondsPresent"
    )(category, level)


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
def generic_multi_indexer(*basis_labels) -> Callable[[str, int], list[tuple[str, ...]]]:
    number_of_levels = 1 if isinstance(basis_labels[0], str) else len(basis_labels[0])
    assert not any(number_of_levels != 1 if isinstance(label, str) else len(label) for label in basis_labels)

    def result(category: str, target_depth: int):
        # if target_depth < number_of_levels:
        if (number_of_levels_to_add := target_depth - number_of_levels - 1) < 0:
            msg = f"Desired number of levels, {target_depth}, is lower than the initial, {number_of_levels}"
            raise ValueError(msg)
        levels_to_add = ["" for _ in range(number_of_levels_to_add)]
        return [(category, label, *levels_to_add) for label in basis_labels]

    return result
