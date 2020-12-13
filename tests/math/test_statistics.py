import numpy as np

from bikipy.math.statistics import feature_scale


def test_feature_scale():
    test_data = (0, 1, 10, 100)
    scaled_test_data = feature_scale(test_data)

    np.testing.assert_almost_equal(scaled_test_data, (0, 1 / 100, 10 / 100, 1.0))
