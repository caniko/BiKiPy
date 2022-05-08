from pathlib import Path

import numpy as np
import pytest

from bikipy.reader.deeplabcut import DeepLabCutReader

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "test_data"
HDF_PATH = EXAMPLES_ROOT / "data_for_angle.h5"


@pytest.fixture
def dlc_test_instance() -> DeepLabCutReader:
    return DeepLabCutReader(df_path=HDF_PATH)


@pytest.mark.parametrize("roi_a, roi_b", (("left_ear", "right_ear"),))
def test_reduce_likelihoods(dlc_test_instance, roi_a, roi_b):
    expected_result = (
        dlc_test_instance.df[(roi_a, "likelihood")].values * dlc_test_instance.df[(roi_b, "likelihood")].values
    )

    np.testing.assert_allclose(expected_result, dlc_test_instance.reduce_likelihoods((roi_a, roi_b)))


@pytest.mark.parametrize("roi", ("left_ear", "right_ear"))
def test_region_of_interest_vs_boolean_index(dlc_test_instance, roi):
    assert dlc_test_instance.region_of_interest_vs_boolean_index[roi] is not None
