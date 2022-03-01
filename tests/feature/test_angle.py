from pathlib import Path

import numpy as np

from bikipy.feature.angle import (
    compute_angles_from_vectors,
)
from bikipy.math.statistics import feature_scale
from bikipy.reader.deeplabcut import DeepLabCutReader

HDF_PATH = Path(__file__).parent.parent.resolve() / "test_data/data_for_angle.h5"


def test_counterclockwise_2d():
    point_1 = ((0, 0), (0, 0), (0, 0))
    point_2 = ((1, 0), (1, 0), (1, 0))
    point_3 = ((2, 0), (1, 1), (1, -1))

    answers = np.array((np.pi, np.pi / 2, 3 * np.pi / 2))

    # AB -> BC
    result = compute_angles_from_vectors(point_1, point_2, point_3)
    np.testing.assert_allclose(result, answers), result

    # CB -> BA
    result = compute_angles_from_vectors(point_3, point_2, point_1)
    np.testing.assert_allclose(result, 2 * np.pi - answers), result

    answers_in_deg = answers * 180 / np.pi
    result = compute_angles_from_vectors(point_1, point_2, point_3, degrees=True)
    np.testing.assert_allclose(result, answers_in_deg), result
