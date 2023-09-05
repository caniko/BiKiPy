from bikipy.utils.pandas import generic_multi_indexer


def test_generic_multi_indexer():
    assert generic_multi_indexer("Displacement", "Median_speed")("MyTest", 5) == [
        ("MyTest", "Displacement", "", "", ""),
        ("MyTest", "Median_speed", "", "", ""),
    ]
