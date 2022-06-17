import numpy as np

from bikipy.feature.motion import (
    displacement_by_frame,
    get_combined_features_from_merged_motion_island_data,
)


def test_get_combined_features_from_merged_motion_island_data():
    fps = 15

    print(
        get_combined_features_from_merged_motion_island_data(
            boolean_index=[False, False, False, True, True, True, True],
            coordinate_sequence=[
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (4, 0),
                (5, 0),
            ],
            fps=fps,
        )
    )


def test_displacement_by_frame():
    assert np.all(
        displacement_by_frame(
            coordinate_sequence=[
                (1, 0),
                (2, 0),
                (3, 0),
                (4, 0),
                (5, 0),
                (4, 0),
                (5, 0),
            ]
        )
    )
