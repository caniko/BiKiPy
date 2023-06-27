from pathlib import Path
from typing import Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.video.VideoClip import VideoClip

from bikipy.perimeter.base import Perimeter
from bikipy.reader.base import Reader


def make_inspection_video(
    video_frames: Iterator[np.ndarray[int, np.dtype[np.uint8]]],
    reader: Reader,
    perimeter_to_boolean_index: Optional[dict[Perimeter, np.ndarray[bool, bool]]] = None,
    label_to_boolean_index: Optional[dict[str, np.ndarray[bool, bool]]] = None,
    label_to_quiver_rays: Optional[dict[str, np.ndarray[float, np.dtype[np.float64]]]] = None,
    output_file_path: Optional[Path] = None,
    codec: str = "hevc_nvenc",
) -> None:
    """
    Create video from trial data.

    :param video_frames: Iterator that stores sequential video frames
    :param reader: Respective reader for video
    :param perimeter_to_boolean_index: K: perimeter; V: Boolean index stating confinement,
        confinement is green otherwise red
    :param label_to_boolean_index:
    :param label_to_quiver_rays:
    :param output_file_path:
    :param codec: ffmpeg codec to use for video encoding. Recommended CPU encoder: mpeg4;
        GPU encoders:
            - Nvidia (<4000-series): hevc_nvenc --new--> "av1_nvenc"
            - AMD: "hevc_amf" --new--> "av1_amf"
            - Intel arc: av1_qsv
    """
    output_file_path = output_file_path or reader.df_path.with_suffix(".mp4")

    if label_to_quiver_rays and not label_to_boolean_index:
        msg = "label_to_boolean_index must be defined for use of label_to_quiver_rays"
        raise ValueError(msg)

    if reader.crop_frames_from_start:
        for i in range(reader.crop_frames_from_start):
            next(video_frames)

    if label_to_boolean_index:
        label_to_boolean_index = {}
    if label_to_quiver_rays:
        label_to_quiver_rays = {}

    labels_to_exclude = frozenset(label_to_boolean_index if label_to_boolean_index else ())

    def make_frame(frame_idx: int):
        fig, ax = plt.subplots()

        ax.imshow(next(video_frames))

        reader.plot_skeleton_in_frame(frame_idx, ax, labels_to_exclude)

        for label, boolean_index in label_to_boolean_index.items():
            is_confinement = boolean_index[frame_idx]
            color = "g" if is_confinement else reader.label_to_plot_color[label]
            if is_confinement and label in label_to_quiver_rays:
                # We draw arrows only when confinement
                ax.quiver(
                    *reader.coordinates_for_plot(label)[frame_idx],
                    *label_to_quiver_rays[label][frame_idx],
                    label=label,
                    color=color,
                )
            else:
                ax.scatter(reader.coordinates_for_plot(label)[frame_idx], c=color, label=label)

        for perimeter, boolean_index in perimeter_to_boolean_index.items():
            perimeter.plot_perimeter_on_ax(ax, color="b" if boolean_index[frame_idx] else "r")

        # Place the legend outside the plot on the right
        ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")

        # Adjust the layout so it fits the legend
        plt.tight_layout()

        return mplfig_to_npimage(fig)

    # Write the video file
    VideoClip(make_frame, duration=reader.number_of_frames).set_fps(reader.video.fps).write_videofile(
        str(output_file_path), codec=codec
    )
