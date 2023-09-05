import numpy as np

from bikipy.feature.motion import displacement_by_frame


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
