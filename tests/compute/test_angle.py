from pathlib import Path
import numpy as np

from kinpy import DeepLabCutReader
from kinpy.compute import angle_over_time, clockwise_2d

HDF_PATH = Path().resolve() / "data_for_angle.h5"


def test_angle_over_time():
    dlcr = DeepLabCutReader.from_hdf(
        HDF_PATH,
        video_res=(1280, 720),
        center_bp=[("left_ear", "right_ear")],
        future_scaling=False,
        min_like=0.95
    )

    tail_base_tip = angle_over_time(
        dlcr, point_a="tail_base", point_b="tail_mid", point_c="tail_tip"
    )

    head_tail = angle_over_time(
        dlcr, point_a="c_left_ear_right_ear", point_b="tail_base", point_c="tail_tip"
    )

    assert tail_base_tip["Angles"].size == dlcr.df.index.size, tail_base_tip["Angles"]

    assert all(tail_base_tip["Accuracy Score"] <= 1.0)
    assert all(head_tail["Accuracy Score"] <= 1.0)

    tail_base_tip_no_nan = tail_base_tip["Angles"][~np.isnan(tail_base_tip["Angles"])]
    head_tail_no_nan = head_tail["Angles"][~np.isnan(head_tail["Angles"])]

    assert all(np.logical_and(
        tail_base_tip_no_nan >= 0, tail_base_tip_no_nan <= 2 * np.pi
    ))
    assert all(np.logical_and(head_tail_no_nan >= 0, head_tail_no_nan <= 2 * np.pi))


def test_clockwise_angle():
    a = np.array([[0, 0]])
    b = np.array([[1, 0]])
    c = np.array([[2, 0]])

    ab_vec = a - b
    bc_vec = b - c

    assert np.isclose(clockwise_2d(ab_vec, bc_vec)[0], np.pi)
