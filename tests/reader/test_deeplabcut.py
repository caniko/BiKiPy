from pathlib import Path

from bikipy.core.video import VideoMetadata
from bikipy.reader.data_with_likelihood import DeepLabCutReader

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "test_data"

HDF_PATH = EXAMPLES_ROOT / "data_for_angle.h5"
CSV_PATH = EXAMPLES_ROOT / "test_tracking.csv"
VIDEO_PATH = EXAMPLES_ROOT / "test_video.mp4"
IMG_PATH = EXAMPLES_ROOT / "test.png"

video = VideoMetadata(manual_resolution=(388, 442), fps=30, meters_per_pixel=[0.94, 1.0])


def test_deeplabcut_reader_from_csv():
    assert DeepLabCutReader(
        df_path=CSV_PATH,
        manual_video=video,

        midpoint_groups={"center_ear": ("left_ear", "right_ear")},
    )


def test_deeplabcut_reader_from_hdf():
    assert DeepLabCutReader(
        df_path=HDF_PATH,
        manual_video=video,

        midpoint_groups={"center_ear": ("left_ear", "right_ear")},
    )


def test_deeplabcut_reader_augmented():
    df = DeepLabCutReader(
        df_path=HDF_PATH,
        manual_video=video,

        midpoint_groups={"center_ear": ("left_ear", "right_ear")},
    )
    assert not df.augmented.empty


def test_deeplabcut_reader_getitem():
    df = DeepLabCutReader(
        df_path=HDF_PATH,
        manual_video=video,

        midpoint_groups={"center_ear": ("left_ear", "right_ear")},
    )
    assert df["center_ear"] is not None
