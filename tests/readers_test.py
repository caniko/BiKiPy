from pathlib import Path

import kinpy


EXAMPLES_ROOT = Path(__file__).resolve().parent / "example_data"

HDF_PATH = EXAMPLES_ROOT / "data_for_angle.h5"
CSV_PATH = EXAMPLES_ROOT / "test_tracking.csv"
VIDEO_PATH = EXAMPLES_ROOT / "test_video.mp4"
IMG_PATH = EXAMPLES_ROOT / "test.png"

RESOLUTION = (1280, 720)


def test_deep_lab_cut_reader():
    from_video_obj = kinpy.DeepLabCutReader.from_video(
        str(VIDEO_PATH),
        future_scaling=True,
        csv_path=CSV_PATH,
        midpoint_pairs=[("left_ear", "right_ear")],
    )

    assert from_video_obj

    from_csv_obj = kinpy.DeepLabCutReader.from_csv(
        str(CSV_PATH),
        RESOLUTION,
        future_scaling=True,
        midpoint_pairs=[("left_ear", "right_ear")],
    )

    assert from_csv_obj

    from_hdf_obj = kinpy.DeepLabCutReader.from_hdf(
        str(HDF_PATH),
        RESOLUTION,
        future_scaling=True,
        midpoint_pairs=[("left_ear", "right_ear")],
    )

    assert from_hdf_obj
