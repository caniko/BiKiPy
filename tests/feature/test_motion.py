from bikipy.feature.motion import get_combined_features_from_merged_motion_island_data


def test_get_combined_features_from_merged_motion_island_data():
    unit_per_pixel = 0.1
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
            units_per_pixel=unit_per_pixel,
            fps=fps,
        )
    )
