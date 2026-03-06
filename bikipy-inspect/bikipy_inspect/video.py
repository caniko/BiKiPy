from __future__ import annotations

import os
from logging import getLogger
from pathlib import Path
from typing import Iterator

import cv2
import matplotlib.pyplot as plt
import numpy as np
from moviepy.video.VideoClip import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

from bikipy_inspect.manifest import InspectionManifest
from bikipy_inspect.plot.perimeter import plot_perimeter_on_ax

logger = getLogger(__name__)

QUIVER_KWARGS = {
    "alpha": 0.60,
    "angles": "xy",
    "units": "xy",
}


def video_frame_iterator(video_path: str | Path) -> Iterator[np.ndarray]:
    """Yield frames from a video file using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def make_inspection_video(
    manifest: InspectionManifest,
    output_path: Path | None = None,
    codec: str = "h264",
    labels_to_plot: list[str] | None = None,
    heuristic_name: str | None = None,
    delete_old: bool = True,
) -> None:
    """Create an inspection video from manifest data with matplotlib overlays.

    Renders coordinates on each video frame with optional:
    - Perimeter outlines (colored by confinement state)
    - Heuristic boolean coloring (green=True, red=False)
    - Quiver plots for ray directions
    """
    if manifest.video.path is None:
        msg = "Video path not set in manifest"
        raise ValueError(msg)

    video_path = Path(manifest.video.path)
    if not video_path.exists():
        msg = f"Video file not found: {video_path}"
        raise FileNotFoundError(msg)

    output_path = output_path or video_path.with_suffix(".inspection.mp4")
    labels = labels_to_plot or manifest.labels
    fps = manifest.video.fps
    duration = manifest.video.duration_seconds
    mpp = manifest.settings.meters_per_pixel

    # Pre-fetch coordinate arrays and boolean indices
    label_coords = {}
    for label in labels:
        if label in manifest.coordinate_columns:
            coords = manifest.get_coordinates(label)
            # Convert metric to pixel coordinates for video overlay
            if mpp > 0:
                coords = coords / mpp
            label_coords[label] = coords

    boolean_index = None
    if heuristic_name:
        boolean_index = manifest.get_boolean_index(heuristic_name)

    frames = video_frame_iterator(video_path)

    _current_frame_idx: int = 0
    _current_frame: np.ndarray | None = None

    def make_frame(next_frame_idx: int):
        nonlocal _current_frame_idx, _current_frame

        if _current_frame is not None and _current_frame_idx == next_frame_idx:
            return _current_frame

        _current_frame_idx = next_frame_idx

        fig, ax = plt.subplots()
        ax.imshow(next(frames))

        # Plot label coordinates as scatter points
        for label, coords in label_coords.items():
            if next_frame_idx >= len(coords):
                continue

            color = manifest.label_colors.get(label, "cyan")
            point = coords[next_frame_idx]

            if boolean_index is not None and next_frame_idx < len(boolean_index):
                marker_color = "lime" if boolean_index[next_frame_idx] else "red"
            else:
                marker_color = color

            ax.scatter(point[0], point[1], color=marker_color, label=label, marker=".", s=30)

        # Draw perimeters
        for p in manifest.perimeters:
            perim_color = "b"
            if boolean_index is not None and next_frame_idx < len(boolean_index):
                perim_color = "g" if boolean_index[next_frame_idx] else "r"
            plot_perimeter_on_ax(ax, p, color=perim_color, meters_per_pixel=mpp, as_pixels=True)

        ax.legend(bbox_to_anchor=(1.01, 0.5), loc="center left")
        plt.tight_layout()

        _current_frame = mplfig_to_npimage(fig)
        plt.close(fig)
        return _current_frame

    if delete_old and output_path.exists():
        os.remove(output_path)

    logger.info(
        "Creating inspection video: %s. Codec: %s. Duration: %.1fs. FPS: %.1f.",
        output_path, codec, duration, fps,
    )

    (
        VideoClip(
            lambda t: make_frame(round(t * fps)),
            duration=duration,
        )
        .set_fps(fps)
        .write_videofile(str(output_path), codec=codec, preset="slower")
    )
