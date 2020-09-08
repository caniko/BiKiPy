from bikipy.utils.statistics import feature_scale


def test_feature_scale():
    test_data = (0, 1, 10, 100)
    scaled_test_data = feature_scale(test_data)

    assert tuple(scaled_test_data) == (0, 1 / 100, 10 / 100, 1), scaled_test_data
