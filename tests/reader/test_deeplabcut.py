from functools import partial
from pathlib import Path

from bikipy.reader.deeplabcut import DeepLabCutReader

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "example_data"

HDF_PATH = EXAMPLES_ROOT / "data_for_angle.h5"
CSV_PATH = EXAMPLES_ROOT / "test_tracking.csv"
VIDEO_PATH = EXAMPLES_ROOT / "test_video.mp4"
IMG_PATH = EXAMPLES_ROOT / "test.png"

PIXEL_RESOLUTION = (388, 442)


def test_deep_lab_cut_reader_from_video():
    partial_dlc = partial(
        DeepLabCutReader.from_video,
        str(VIDEO_PATH),
        future_scaling=True,
        midpoint_groups=[("left_ear", "right_ear")],
    )

    assert partial_dlc(csv_path=CSV_PATH)
    assert partial_dlc(hdf_path=HDF_PATH)

    assert partial_dlc(hdf_path=HDF_PATH).pixel_resolution == PIXEL_RESOLUTION


def test_deep_lab_cut_reader_from_csv():
    assert DeepLabCutReader.from_csv(
        str(CSV_PATH),
        PIXEL_RESOLUTION,
        future_scaling=True,
        midpoint_groups=[("left_ear", "right_ear")],
    )


def test_deep_lab_cut_reader_from_hdf():
    assert DeepLabCutReader.from_hdf(
        str(HDF_PATH),
        PIXEL_RESOLUTION,
        future_scaling=True,
        midpoint_groups=[("left_ear", "right_ear")],
    )
