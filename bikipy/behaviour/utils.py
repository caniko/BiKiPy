from typing import Iterable

import numpy as np
from pydantic_numpy.typing import NpNDArray


def unique_with_counts_zipped(array: NpNDArray):
    return zip(*np.unique(array, return_counts=True))


def feature_2d_multi_indexer(feature: str, groups: Iterable[str]):
    return [(str(feature), str(group)) for group in groups]


def blanket_experiment_label_generator(experiment_label: str) -> set[str]:
    return {
        experiment_label,
        f"{experiment_label}-enclosed",
        f"blanket-{experiment_label}",
        f"generic-{experiment_label}",
    }


def blanket_enclosed_experiment_label_generator(experiment_label: str) -> set[str]:
    result = blanket_experiment_label_generator(experiment_label)
    result.add(f"{experiment_label.capitalize()}Enclosed")
    return result
