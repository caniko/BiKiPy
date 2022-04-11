from pathlib import Path

import numpy as np

from bikipy.feature.angle import compute_angles_from_vectors

HDF_PATH = Path(__file__).parent.parent.resolve() / "test_data/data_for_angle.h5"


def test_compute_angles_from_vectors():
    point_1 = np.array(((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))
    point_2 = np.array(((1.0, 0.0), (1.0, 0.0), (1.0, 0.0)))
    point_3 = np.array(((2.0, 0.0), (1.0, 1.0), (1.0, -1.0)))

    # Inner
    answers = np.array((0.0, -np.pi / 2.0, np.pi / 2.0))

    # AB -> BC
    result = compute_angles_from_vectors(point_1, point_2, point_3, method="inner")
    np.testing.assert_allclose(result, answers)

    # CB -> BA
    result = compute_angles_from_vectors(point_3, point_2, point_1, method="inner")
    np.testing.assert_allclose([result[i] for i in (0, 2, 1)], answers)

    answers_in_deg = np.rad2deg(answers)  # * 180 / np.pi
    result = compute_angles_from_vectors(point_1, point_2, point_3, method="inner", degrees=True)
    np.testing.assert_allclose(result, answers_in_deg)

    answers = np.array((0.0, 3.0 * np.pi / 2.0, np.pi / 2.0))

    # AB -> BC
    result = compute_angles_from_vectors(point_1, point_2, point_3, method="counterclockwise")
    np.testing.assert_allclose(result, answers)

    # CB -> BA
    result = compute_angles_from_vectors(point_3, point_2, point_1, method="counterclockwise")
    np.testing.assert_allclose([result[i] for i in (0, 2, 1)], answers)

    answers_in_deg = np.rad2deg(answers)  # * 180 / np.pi
    result = compute_angles_from_vectors(point_1, point_2, point_3, method="counterclockwise", degrees=True)
    np.testing.assert_allclose(result, answers_in_deg)
