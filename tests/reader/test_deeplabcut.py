from functools import partial
from pathlib import Path

from bikipy.reader.deeplabcut import DeepLabCutReader

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "test_data"

HDF_PATH = EXAMPLES_ROOT / "data_for_angle.h5"
CSV_PATH = EXAMPLES_ROOT / "test_tracking.csv"
VIDEO_PATH = EXAMPLES_ROOT / "test_video.mp4"
IMG_PATH = EXAMPLES_ROOT / "test.png"

PIXEL_RESOLUTION = (388, 442)


def test_deep_lab_cut_reader_from_csv():
    assert DeepLabCutReader(
        df_path=CSV_PATH,
        future_scaling=True,
        midpoint_groups={"center_eye": ("left_ear", "right_ear")},
    )


def test_deep_lab_cut_reader_from_hdf():
    assert DeepLabCutReader(
        df_path=HDF_PATH,
        future_scaling=True,
        midpoint_groups={"center_eye": ("left_ear", "right_ear")},
    )
