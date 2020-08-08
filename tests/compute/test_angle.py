from pathlib import Path
import numpy as np

from bikipy.compute.angle import dlc_angle_over_time, inner_clockwise_angel_2d
from bikipy import DeepLabCutReader

HDF_PATH = Path(__file__).parent.parent.resolve() / "example_data/data_for_angle.h5"


def test_angle_over_time():
    test_data = DeepLabCutReader.from_hdf(
        HDF_PATH,
        video_res=(1280, 720),
        midpoint_groups=[("left_ear", "right_ear")],
        future_scaling=False,
        min_likelihood=0.95,
    )

    tail_base_tip = dlc_angle_over_time(
        test_data.df, point_a_name="tail_base", point_b_name="tail_mid", point_c_name="tail_tip"
    )

    head_tail = dlc_angle_over_time(
        test_data.df,
        point_a_name="mid-left_ear-right_ear",
        point_b_name="tail_base",
        point_c_name="tail_tip",
    )

    assert tail_base_tip["Angle"].size == test_data.df.index.size, tail_base_tip[
        "Angle"
    ]

    assert all(tail_base_tip["Likelihood"] <= 1.0)
    assert all(head_tail["Likelihood"] <= 1.0)

    tail_base_tip_no_nan = tail_base_tip["Angle"][~np.isnan(tail_base_tip["Angle"])]
    head_tail_no_nan = head_tail["Angle"][~np.isnan(head_tail["Angle"])]

    assert all(
        np.logical_and(tail_base_tip_no_nan >= 0, tail_base_tip_no_nan <= 2 * np.pi)
    ), tail_base_tip_no_nan
    assert all(
        np.logical_and(head_tail_no_nan >= 0, head_tail_no_nan <= 2 * np.pi)
    ), tail_base_tip_no_nan


def test_clockwise_2d():
    def three_point_vector_path(point_1, point_2, point_3, answer):
        point_1 = np.asanyarray(point_1)
        point_2 = np.asanyarray(point_2)
        point_3 = np.asanyarray(point_3)

        vector_1_2 = point_2 - point_1
        vector_2_3 = point_3 - point_2

        result = inner_clockwise_angel_2d(vector_1_2, vector_2_3)[0]
        assert np.isclose(result, answer), result

        result = inner_clockwise_angel_2d(vector_2_3, vector_1_2)[0]
        assert np.isclose(result, 2 * np.pi - answer), result

    three_point_vector_path(((0, 0),), ((1, 0),), ((2, 0),), np.pi)

    three_point_vector_path(((0, 0),), ((1, 0),), ((1, 1),), np.pi / 2)

    three_point_vector_path(((0, 0),), ((1, 0),), ((1, -1),), 3 * np.pi / 2)


test_angle_over_time()
