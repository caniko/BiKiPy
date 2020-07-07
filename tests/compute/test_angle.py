from pathlib import Path
import numpy as np

from kinpy import DeepLabCutReader
from kinpy.compute import angle_over_time, clockwise_2d

HDF_PATH = Path(__file__).parent.parent.resolve() / "example_data/data_for_angle.h5"


def test_angle_over_time():
    dlcr = DeepLabCutReader.from_hdf(
        HDF_PATH,
        video_res=(1280, 720),
        center_bp=[("left_ear", "right_ear")],
        future_scaling=False,
        min_like=0.95,
    )

    tail_base_tip = angle_over_time(
        dlcr.df, point_a="tail_base", point_b="tail_mid", point_c="tail_tip"
    )

    head_tail = angle_over_time(
        dlcr.df, point_a="c_left_ear_right_ear", point_b="tail_base", point_c="tail_tip"
    )

    assert tail_base_tip["Angles"].size == dlcr.df.index.size, tail_base_tip["Angles"]

    assert all(tail_base_tip["Accuracy Score"] <= 1.0)
    assert all(head_tail["Accuracy Score"] <= 1.0)

    tail_base_tip_no_nan = tail_base_tip["Angles"][~np.isnan(tail_base_tip["Angles"])]
    head_tail_no_nan = head_tail["Angles"][~np.isnan(head_tail["Angles"])]

    assert all(
        np.logical_and(tail_base_tip_no_nan >= 0, tail_base_tip_no_nan <= 2 * np.pi)
    )
    assert all(np.logical_and(head_tail_no_nan >= 0, head_tail_no_nan <= 2 * np.pi))


def test_clockwise_2d():
    def three_point_vector_path(
        point_1,
        point_2,
        point_3,
        answer
    ):
        point_1 = np.asarray(point_1)
        point_2 = np.asarray(point_2)
        point_3 = np.asarray(point_3)

        vector_1_2 = point_2 - point_1
        vector_2_3 = point_3 - point_2

        result = clockwise_2d(vector_1_2, vector_2_3)[0]
        assert np.isclose(result, answer), result

        result = clockwise_2d(vector_2_3, vector_1_2)[0]
        assert np.isclose(result, 2*np.pi - answer), result

    three_point_vector_path(
        ((0, 0),), ((1, 0),), ((2, 0),), np.pi
    )

    three_point_vector_path(
        ((0, 0),), ((1, 0),), ((1, 1),), np.pi / 2
    )

    three_point_vector_path(
        ((0, 0),), ((1, 0),), ((1, -1),), 3*np.pi / 2
    )
