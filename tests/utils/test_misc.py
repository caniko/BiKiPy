from bikipy.utils.misc import generic_multi_indexer


def test_generic_multi_indexer():
    multi_index = generic_multi_indexer("Displacement", "Median_speed")("MyTest", 5)
    1
