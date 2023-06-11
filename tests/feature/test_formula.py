import numpy as np

from bikipy.feature.qualia.formula import parse_heuristic_formula


def test_parse_heuristic_formula():
    assert np.all(
        parse_heuristic_formula("x | ~y", {"x": np.array([True, False]), "y": np.array([False, False])})
        == np.array([True, True])
    )
