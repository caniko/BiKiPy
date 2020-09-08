from pathlib import Path
import numpy as np

from bikipy.utils.statistics import feature_scale
from bikipy.readers import DeepLabCutReader
from bikipy.compute.angle import (
    compute_angles_from_vectors,
    dlc_compute_angles_from_vectors,
)

HDF_PATH = Path(__file__).parent.parent.resolve() / "example_data/data_for_angle.h5"


def test_compute_angles_from_vectors():
    test_data = DeepLabCutReader.from_hdf(
        HDF_PATH,
        video_res=(1280, 720),
        midpoint_groups=[("left_ear", "right_ear")],
        future_scaling=False,
        min_likelihood=0.95,
    )

    tail_base_tip = dlc_compute_angles_from_vectors(
        test_data.df,
        point_a_name="tail_base",
        point_b_name="tail_mid",
        point_c_name="tail_tip",
    )

    head_tail = dlc_compute_angles_from_vectors(
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
    point_1 = ((0, 0), (0, 0), (0, 0))
    point_2 = ((1, 0), (1, 0), (1, 0))
    point_3 = ((2, 0), (1, 1), (1, -1))

    answers = np.array((np.pi, np.pi / 2, 3 * np.pi / 2))

    # AB -> BC
    result = compute_angles_from_vectors(point_1, point_2, point_3)
    assert np.allclose(result, answers), result

    # CB -> BA
    result = compute_angles_from_vectors(point_3, point_2, point_1)
    assert np.allclose(result, 2 * np.pi - answers), result

    feature_scaled_answers = feature_scale(answers)
    result = compute_angles_from_vectors(
        point_1, point_2, point_3,
        feature_scale_data=True
    )
    assert np.allclose(result, feature_scaled_answers), result

    answers_in_deg = answers * 180 / np.pi
    result = compute_angles_from_vectors(
        point_1, point_2, point_3,
        degrees=True
    )
    assert np.allclose(result, answers_in_deg), result
