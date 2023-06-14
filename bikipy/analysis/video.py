from pathlib import Path
from typing import Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.VideoClip import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

from bikipy.perimeter.base import Perimeter
from bikipy.reader.base import Reader


def make_inspection_video(
    video_frames: Iterator[np.ndarray[int, np.dtype[np.uint8]]],
    reader: Reader,
    perimeter_to_boolean_index: Optional[dict[Perimeter, np.ndarray[bool, bool]]] = None,
    output_file_path: Optional[Path] = None,
):
    output_file_path = output_file_path or reader.df_path.with_suffix(".mp4")

    if reader.crop_time_seconds and not reader.crop_from_end:
        for i in range(reader.crop_frames):
            next(video_frames)

    def make_frame(frame_idx: int):
        fig, ax = plt.subplots()

        frame = next(video_frames)
        ax.imshow(frame)

        reader.plot_skeleton_in_frame(frame_idx, ax)

        for perimeter, boolean_index in perimeter_to_boolean_index.items():
            perimeter.plot_perimeter_on_ax(ax, color="b" if boolean_index[frame_idx] else "r")

        return mplfig_to_npimage(fig)

    # Create a video clip
    clip = VideoClip(make_frame, duration=reader.number_of_frames)

    # Write the video file
    clip.set_fps(reader.video.fps).write_videofile(str(output_file_path), codec="libx264")
