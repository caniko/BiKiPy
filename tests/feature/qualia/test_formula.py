import numpy as np
import pytest

from bikipy.feature.qualia.physical_object.merge_parser import (
    parse_heuristic_merge_equation,
)


def test_parse_heuristic_merge_equation():
    alias_to_qualia_analysis = {
        "A": np.array([True, False, True, False]),
        "B": np.array([True, True, False, False]),
        "C": np.array([False, True, True, False]),
    }

    # Simple AND operation
    assert np.array_equal(
        parse_heuristic_merge_equation("A and B", alias_to_qualia_analysis), np.array([True, False, False, False])
    )

    # Simple OR operation
    assert np.array_equal(
        parse_heuristic_merge_equation("A or B", alias_to_qualia_analysis), np.array([True, True, True, False])
    )

    # Simple NOT operation
    assert np.array_equal(
        parse_heuristic_merge_equation("not A", alias_to_qualia_analysis), np.array([False, True, False, True])
    )

    # Testing OR operation with "|"
    assert np.array_equal(
        parse_heuristic_merge_equation("A | B", alias_to_qualia_analysis), np.array([True, True, True, False])
    )

    # Testing AND operation with "&"
    assert np.array_equal(
        parse_heuristic_merge_equation("A & B", alias_to_qualia_analysis), np.array([True, False, False, False])
    )

    # Testing NOT operation with "~"
    assert np.array_equal(
        parse_heuristic_merge_equation("~A", alias_to_qualia_analysis), np.array([False, True, False, True])
    )

    # Testing OR with more than 2 components
    assert np.array_equal(
        parse_heuristic_merge_equation("A or B or C", alias_to_qualia_analysis), np.array([True, True, True, False])
    )

    # Testing AND with more than 2 components
    assert np.array_equal(
        parse_heuristic_merge_equation("A and B and C", alias_to_qualia_analysis),
        np.array([False, False, False, False]),
    )

    # Testing OR with more than 2 components, NOT in mid
    assert np.array_equal(
        parse_heuristic_merge_equation("A or ~B or C", alias_to_qualia_analysis), np.array([True, True, True, True])
    )

    # Testing AND with more than 2 components, NOT in mid
    assert np.array_equal(
        parse_heuristic_merge_equation("A and ~B and C", alias_to_qualia_analysis),
        np.array([False, False, True, False]),
    )

    # Invalid variable name
    with pytest.raises(Exception):
        parse_heuristic_merge_equation("D and B", alias_to_qualia_analysis)

    # Invalid syntax
    with pytest.raises(Exception):
        parse_heuristic_merge_equation("A andor B", alias_to_qualia_analysis)
