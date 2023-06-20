import pytest
import numpy as np

from bikipy.feature.qualia.physical_object.merge_parser import parse_heuristic_merge_equation


@pytest.fixture
def setup_data():
    alias_to_heuristic_result = {
        "A": [np.array([True, True, False, False])],
        "B": [np.array([True, False, True, False])],
        "C": [np.array([False, True, True, True])],
    }
    return alias_to_heuristic_result


def test_parse_heuristic_merge_equation_or(setup_data):
    result = parse_heuristic_merge_equation("A or B", setup_data)
    np.testing.assert_array_equal(result, [np.array([True, True, True, False])])


def test_parse_heuristic_merge_equation_and(setup_data):
    result = parse_heuristic_merge_equation("A and B", setup_data)
    np.testing.assert_array_equal(result, [np.array([True, False, False, False])])


def test_parse_heuristic_merge_equation_not(setup_data):
    result = parse_heuristic_merge_equation("not A", setup_data)
    np.testing.assert_array_equal(result, [np.array([False, False, True, True])])


def test_parse_heuristic_merge_equation_complex(setup_data):
    result = parse_heuristic_merge_equation("A and not B or C", setup_data)
    np.testing.assert_array_equal(result, [np.array([False, True, True, True])])
