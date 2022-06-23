import numpy as np

from bikipy.core.video import VideoMetadata

video_test_a = VideoMetadata(
    manual_recording_resolution=[1920, 1080],
    manual_fps=30,
)
video_test_b = VideoMetadata(
    meters_per_pixel=[1, 1],
    manual_recording_resolution=[1920, 1080],
)

INCONGRUENT_RESOLUTION = np.array([1000, 1080], dtype=np.int16)
video_test_c = VideoMetadata(
    meters_per_pixel=[1, 1],
    manual_recording_resolution=INCONGRUENT_RESOLUTION,
)


def test_video_metadata_join():
    video_test_complete = video_test_a + video_test_b

    assert np.any(video_test_complete.recording_resolution) and np.any(video_test_complete.meters_per_pixel)


def test_video_metadata_comparison():
    assert video_test_a != video_test_b
    assert video_test_a & video_test_b


def test_video_metadata_incongruent():
    video_test_complete = video_test_a + video_test_b
    dirty_video_test = VideoMetadata.join(video_test_c, video_test_complete, ignore_incongruency=True)
    assert np.all(dirty_video_test.recording_resolution == INCONGRUENT_RESOLUTION)
