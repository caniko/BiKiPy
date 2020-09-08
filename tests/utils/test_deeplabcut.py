import pandas as pd
import numpy as np

from bikipy.utils.deeplabcut import reduce_likelihoods, get_region_of_interest_data


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
