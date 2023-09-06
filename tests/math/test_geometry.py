from bikipy.math.geometry import expand_rectangle


def _check_all_vectors_exist(array1, array2):
    # Convert arrays to sets of tuples for faster look-up
    set1 = set([tuple(row) for row in array1])
    set2 = set([tuple(row) for row in array2])

    # Check if all vectors in set1 exist in set2
    return all(vec in set2 for vec in set1)


def test_expand_rectangle():
    result = expand_rectangle(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), x_offset=1, y_offset=1, y_inverted=False
    )
    expected = ((-1.0, 2.0), (2.0, 2.0), (2.0, -1.0), (-1.0, -1.0))
    assert _check_all_vectors_exist(result, expected)
