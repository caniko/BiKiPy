from pathlib import Path

path_to_this_directory = Path(__file__).parent

EXPERIMENT_KWARGS = {
    "ray_start_point_label": "center_eye",
    "ray_travel_direction_point_label": "nose",
    "object_tracking_label_for_kinematics": "center_eye",
    "perimeter_border_normal_meters": 0.02,
    "manual_reader_kwargs": {"init_from": "parquet", "midpoint_groups": {"center_eye": ["left_ear", "right_ear"]}},
    "trial_id_to_keyword_arguments": {
        1: {
            "framewise_coordinates_path": path_to_this_directory / "test_tracking.csv",
            "animal_id": 1,
        }
    },
}
