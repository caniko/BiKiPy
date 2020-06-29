from pathlib import Path

import kinpy


EXAMPLES_ROOT = Path().resolve() / "example_data"

CSV_PATH = EXAMPLES_ROOT / "test_tracking.csv"
VIDEO_PATH = EXAMPLES_ROOT / "test_video.mp4"
IMG_PATH = EXAMPLES_ROOT / "test.png"


def test_deep_lab_cut_reader():
    from_video_obj = kinpy.DeepLabCutReader.from_video(
        str(VIDEO_PATH),
        future_scaling=True,
        csv_path=CSV_PATH,
        center_bp=[("left_ear", "right_ear")],
    )

    assert from_video_obj

    assert from_video_obj.remove_flicks_hv()
