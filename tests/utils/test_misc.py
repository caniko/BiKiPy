from bikipy.utils.collection_utils import generic_multi_indexer


def test_generic_multi_indexer():
    multi_index = generic_multi_indexer("Displacement", "Median_speed")("MyTest", 5)
    1
