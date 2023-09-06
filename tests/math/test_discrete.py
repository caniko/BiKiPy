import numpy as np

from bikipy.math.discrete import (
    tolerance_modeled_boolean_index_truth_sequence_start_end_length,
)


def test_get_combined_features_from_merged_motion_island_data():
    tolerance_modeled_boolean_index_truth_sequence_start_end_length(
        boolean_index=np.array((False, False, False, True, True, True, True)),
        fps=15,
    )
