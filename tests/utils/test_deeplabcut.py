from pathlib import Path

import numpy as np
import pytest

from bikipy.reader.deeplabcut import DeepLabCutReader
from bikipy.utils.deeplabcut import get_region_of_interest_data, reduce_likelihoods

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "test_data"
HDF_PATH = EXAMPLES_ROOT / "data_for_angle.h5"


@pytest.fixture
def dlc_test_instance():
    return DeepLabCutReader.from_hdf(str(HDF_PATH))


@pytest.mark.parametrize("roi_a, roi_b", (("left_ear", "right_ear"),))
def test_reduce_likelihoods(dlc_test_instance, roi_a, roi_b):
    expected_result = np.expand_dims(
        dlc_test_instance.df[(roi_a, "likelihood")].values
        * dlc_test_instance.df[(roi_b, "likelihood")].values,
        axis=1,
    )

    np.testing.assert_allclose(
        expected_result, reduce_likelihoods(dlc_test_instance.df, (roi_a, roi_b))
    )


@pytest.mark.parametrize("roi", ("left_ear", "right_ear"))
def test_get_region_of_interest_data(dlc_test_instance, roi):
    assert np.any(get_region_of_interest_data(dlc_test_instance.df, roi))
