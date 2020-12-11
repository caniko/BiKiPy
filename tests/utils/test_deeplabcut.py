import numpy as np
import pandas as pd

from bikipy.utils.deeplabcut import get_region_of_interest_data, reduce_likelihoods

test_data = {
    ("Left_Ear", "x"): np.random.random(5),
    ("Left_Ear", "x"): np.random.random(5),
    ("Left_Ear", "likelihood"): np.random.random(5),
    ("Right_Ear", "x"): np.random.random(5),
    ("Right_Ear", "x"): np.random.random(5),
    ("Right_Ear", "likelihood"): np.random.random(5),
}

test_df = pd.DataFrame.from_dict(test_data)


def test_reduce_likelihoods():
    expected_result = np.expand_dims(
        test_data[("Left_Ear", "likelihood")] * test_data[("Right_Ear", "likelihood")],
        axis=1,
    )
    reduced_likelihoods = reduce_likelihoods(test_df, ("Left_Ear", "Right_Ear"))

    assert np.allclose(expected_result, reduced_likelihoods)
