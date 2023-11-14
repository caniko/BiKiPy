import unittest
import numpy as np

from bikipy.math.high_velocity import high_velocity_removal


# Assuming the high_velocity_removal function is defined here or imported

class TestHighVelocityRemoval(unittest.TestCase):

    def test_basic_functionality(self):
        # Test the basic functionality with a simple case
        position_array = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0], [9.0, 12.0]])
        max_meters_per_frame = 5.0
        expected_result = np.array([[0.0, 0.0], [3.0, 4.0], [np.nan, np.nan], [9.0, 12.0]])
        result = high_velocity_removal(position_array, max_meters_per_frame)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_no_high_velocity(self):
        # Test with no points exceeding the velocity threshold
        position_array = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        max_meters_per_frame = 5.0
        expected_result = position_array
        result = high_velocity_removal(position_array, max_meters_per_frame)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_all_high_velocity(self):
        # Test with all points exceeding the velocity threshold
        position_array = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [2.0, 2.0]])
        max_meters_per_frame = 5.0
        expected_result = np.array([[0.0, 0.0], [np.nan, np.nan], [np.nan, np.nan], [2.0, 2.0]])
        result = high_velocity_removal(position_array, max_meters_per_frame)
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_edge_cases(self):
        # Test edge cases, such as an empty array or a single-point array
        position_array = np.array([], dtype=float)
        max_meters_per_frame = 5.0
        expected_result = np.array([], dtype=float)
        result = high_velocity_removal(position_array, max_meters_per_frame)
        np.testing.assert_array_almost_equal(result, expected_result)

        position_array = np.array([[0.0, 0.0]])
        expected_result = np.array([[0.0, 0.0]])
        result = high_velocity_removal(position_array, max_meters_per_frame)
        np.testing.assert_array_almost_equal(result, expected_result)
