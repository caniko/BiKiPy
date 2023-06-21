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

    # Test 1: Simple AND operation
    assert np.array_equal(
        parse_heuristic_merge_equation("A and B", alias_to_qualia_analysis), np.array([True, False, False, False])
    )

    # Test 2: Simple OR operation
    assert np.array_equal(
        parse_heuristic_merge_equation("A or B", alias_to_qualia_analysis), np.array([True, True, True, False])
    )

    # Test 3: Simple NOT operation
    assert np.array_equal(
        parse_heuristic_merge_equation("not A", alias_to_qualia_analysis), np.array([False, True, False, True])
    )

    # Test 4: Testing OR operation with "|"
    assert np.array_equal(
        parse_heuristic_merge_equation("A | B", alias_to_qualia_analysis), np.array([True, True, True, False])
    )

    # Test 5: Testing AND operation with "&"
    assert np.array_equal(
        parse_heuristic_merge_equation("A & B", alias_to_qualia_analysis), np.array([True, False, False, False])
    )

    # Test 6: Testing NOT operation with "~"
    assert np.array_equal(
        parse_heuristic_merge_equation("~A", alias_to_qualia_analysis), np.array([False, True, False, True])
    )

    # Test 7: Invalid variable name
    with pytest.raises(Exception):
        parse_heuristic_merge_equation("D and B", alias_to_qualia_analysis)

    # Test 8: Invalid syntax
    with pytest.raises(Exception):
        parse_heuristic_merge_equation("A andor B", alias_to_qualia_analysis)
