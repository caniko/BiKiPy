import os
from logging import getLogger
from pathlib import Path
from typing import Iterator, Optional

import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.video.VideoClip import VideoClip
from pydantic_numpy.typing import NpNDArrayBool, NpNDArrayFp64, NpNDArrayUint8

from bikipy.perimeter.base import BasePerimeter
from bikipy.reader.base import BaseReader

logger = getLogger(__file__)


def make_inspection_video(
    video_frames: Iterator[NpNDArrayUint8],
    reader: BaseReader,
    perimeter_to_boolean_index: Optional[dict[BasePerimeter, NpNDArrayBool]] = None,
    label_to_confinement_boolean_index: Optional[dict[str, NpNDArrayBool]] = None,
    label_to_quiver_rays: Optional[dict[str, NpNDArrayFp64]] = None,
    revert_crop: bool = False,
    output_file_path: Optional[Path] = None,
    frames_on_the_fly: bool = True,
    codec: str = "h264",
    delete_old: bool = True,
) -> None:
    """
    Create video from trial data.

    :param video_frames: Iterator that stores sequential video frames
    :param reader: Respective reader for video
    :param perimeter_to_boolean_index: K: perimeter; V: Boolean index stating confinement,
        confinement is green otherwise red
    :param label_to_confinement_boolean_index:
    :param label_to_quiver_rays:
    :param output_file_path:
    :param codec: ffmpeg codec to use for video encoding. Recommended CPU encoder: mpeg4;
        GPU encoders:
            - Nvidia (<4000-series): hevc_nvenc --new--> "av1_nvenc"
            - AMD: "hevc_amf" --new--> "av1_amf"
            - Intel arc: av1_qsv
    """
    output_file_path = output_file_path or reader.df_path.with_suffix(".mp4")

    if label_to_quiver_rays and not label_to_confinement_boolean_index:
        msg = "label_to_confinement_boolean_index must be defined for use of label_to_quiver_rays"
        raise ValueError(msg)

    if reader.reader.crop_frames_slice.start > 0:
        for _ in range(reader.crop_frames_slice.start):
            next(video_frames)

    if not label_to_confinement_boolean_index:
        label_to_confinement_boolean_index = {}
    if not label_to_quiver_rays:
        label_to_quiver_rays = {}

    labels_to_exclude = frozenset(label_to_confinement_boolean_index if label_to_confinement_boolean_index else ())

    _current_frame_idx: int = 0
    _current_frame: NpNDArrayUint8 | None = None

    def make_frame(next_frame_idx: int):
        nonlocal _current_frame_idx, _current_frame

        if _current_frame is not None and _current_frame_idx == next_frame_idx:
            return _current_frame

        _current_frame_idx = next_frame_idx

        fig, ax = plt.subplots()

        ax.imshow(next(video_frames))
        reader.plot_skeleton_in_frame(next_frame_idx, ax, labels_to_exclude, revert_crop)

        for label, confinement_boolean_index in label_to_confinement_boolean_index.items():
            color = "g" if confinement_boolean_index[next_frame_idx] else reader.label_to_plot_color[label]
            if label in label_to_quiver_rays:
                # We draw arrows only when confinement
                ax.quiver(
                    *reader.coordinates_for_plot(label, with_resize=False)[next_frame_idx],
                    *label_to_quiver_rays[label][next_frame_idx],
                    label=label,
                    color=color,
                )
            else:
                ax.scatter(*reader.coordinates_for_plot(label, with_resize=False)[next_frame_idx], c=color, label=label)

        for perimeter, confinement_boolean_index in perimeter_to_boolean_index.items():
            perimeter.plot_perimeter_on_ax(
                ax,
                color="b" if confinement_boolean_index[next_frame_idx] else "r",
                coordinates_as_pixels=True,
                with_resize=False,
                x_pixel_offset=-reader.x_axis_crop_end_point,
                y_pixel_offset=-reader.y_axis_crop_end_point,
            )

        # Place the legend outside the plot on the right
        ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")

        # Adjust the layout so it fits the legend
        plt.tight_layout()

        _current_frame = mplfig_to_npimage(fig)
        plt.close()

        return _current_frame

    if delete_old and output_file_path.exists():
        os.remove(output_file_path)

    print(
        f"Creating video from trial data: {output_file_path}."
        f"Frames on the fly: {frames_on_the_fly}."
        f"Codec: {codec}."
        f"Duration: {reader.total_duration}."
        f"FPS: {reader.video_for_computation().fps}."
    )

    if frames_on_the_fly:
        (
            VideoClip(
                lambda t: make_frame(round(t * reader.video_for_computation().fps)),
                duration=reader.trial_length_seconds,
            )
            .set_fps(reader.video_for_computation().fps)
            .write_videofile(str(output_file_path), codec=codec, preset="slower")
        )

    return

    # not frames_on_the_fly; not implemented

    # with ProcessPoolExecutor() as executor:
    #     frame_features = [executor.submit(make_frame, i) for i in range(reader.number_of_frames)]
    #
    # try:
    #     ImageSequenceClip(
    #         [frame_feature.result() for frame_feature in frame_features], fps=reader.video_for_computation().fps
    #     ).write_videofile(output_file_path, codec=codec)
    # except Exception as e:
    #     if output_file_path.exists():
    #         os.remove(output_file_path)
    #     raise e
