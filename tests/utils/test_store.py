from bikipy.utils.store import RangeDict


def test_range_dict():
    test_obj = RangeDict({3: "Hello"})
    assert test_obj

    test_obj[7] = "world!"
    assert f"{test_obj[5]} {test_obj[9]}" == "Hello world!"

    test_obj[1] = "Zen:"
    assert f"{test_obj[2]} {test_obj[5]} {test_obj[9]}" == "Zen: Hello world!"
