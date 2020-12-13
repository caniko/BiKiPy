from pathlib import Path

import numpy as np
import pytest

from bikipy.reader.deeplabcut import DeepLabCutReader
from bikipy.utils.deeplabcut import get_region_of_interest_data, reduce_likelihoods

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "example_data"
HDF_PATH = EXAMPLES_ROOT / "data_for_angle.h5"


@pytest.fixture
def dlc_test_instance():
    return DeepLabCutReader.from_hdf(str(HDF_PATH))


def test_reduce_likelihoods(dlc_test_instance):
    expected_result = np.expand_dims(
        dlc_test_instance.df[("left_ear", "likelihood")].values
        * dlc_test_instance.df[("right_ear", "likelihood")].values,
        axis=1,
    )

    reduced_likelihoods = reduce_likelihoods(
        dlc_test_instance.df, ("left_ear", "right_ear")
    )

    np.testing.assert_allclose(expected_result, reduced_likelihoods)


def test_get_region_of_interest_data(dlc_test_instance):
    assert np.any(get_region_of_interest_data(dlc_test_instance.df, "left_ear"))
