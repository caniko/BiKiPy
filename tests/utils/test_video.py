from pathlib import Path

import numpy as np

from bikipy.utils.video import get_video_data

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "test_data"
VIDEO_PATH = EXAMPLES_ROOT / "test_video.mp4"

PIXEL_RESOLUTION = (388, 442)


def test_get_video_data():
    _, width, heigh, fps = get_video_data(VIDEO_PATH)

    np.testing.assert_almost_equal(PIXEL_RESOLUTION, (width, heigh))
    np.testing.assert_almost_equal(30.0, fps)
