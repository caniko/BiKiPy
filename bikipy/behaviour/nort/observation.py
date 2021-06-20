"""
2D kinematic filters, 3D not supported.
"""
import os
from logging import getLogger
from pathlib import Path, PurePath
from typing import Any, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bikipy.feature.angle import counterclockwise_angel_2d, inner_angle
from bikipy.feature.motion import attention_per_frame
from bikipy.math.point_in_polygon import points_in_parallelogram
from bikipy.math.vector import closest_line_to_point, unit_vector
from bikipy.perimeter.base import PolygonalPerimeter
from bikipy.utils.misc import read_image

logger = getLogger(__name__)
SCATTER_ALPHA = 0.6


def location_filter(
    nort_object,
    nose: Sequence[Sequence[float]],
    torso: Sequence[Sequence[float]],
    perimeter_border_normal_pixel_magnitude: float,
    inspect: bool = False,
    inspection_ax: Any = None,
) -> Sequence[bool]:
    """
    Filter with basis in proximity rules to NORT object.

    The nose has to be in front of object, while the torso is outside of the object.

    Parameters
    ----------
    nort_object
    nose
        Nose cartesian coordinate location sequence
    torso
        Torso (center) cartesian coordinate location sequence
    perimeter_border_normal_pixel_magnitude
        The magnitude of the normal between the perimeter and the border given in pixels
    inspect
        If True, generate and view an analytics of the resulting filter
    inspection_ax
        matplotlib Axes that the inspection plots will (optionally) be saved in

    Returns
    -------

    """
    # Remove nose points that aren't inside the perimeter
    nose = np.asarray(nose)

    nort_object_border = nort_object.border(perimeter_border_normal_pixel_magnitude)

    nose_within_border = points_in_parallelogram(
        nort_object_border.perimeter_corners[1],
        nort_object_border.perimeter_corners[0],
        nort_object_border.perimeter_corners[2],
        nose,
    )
    torso_outside_polygon = np.logical_not(
        points_in_parallelogram(
            nort_object.perimeter_corners[1],
            nort_object.perimeter_corners[0],
            nort_object.perimeter_corners[2],
            torso,
        )
    )

    # Find states where the nose is within perimeter while the torso is not over object
    result = nose_within_border & torso_outside_polygon

    if inspection_ax is not None or inspect:
        if inspection_ax is None:
            sns.set_theme(style="darkgrid")
            fig, ax = plt.subplots()
        else:
            ax = inspection_ax

        ax.set_title("Location filter")
        nort_object.plot(ax=ax)

        not_result = ~result
        ax.scatter(
            *nose[nose_within_border & not_result].T,
            alpha=SCATTER_ALPHA,
            label="Nose valid, invalid torso",
        )
        ax.scatter(
            *nose[torso_outside_polygon & not_result].T,
            alpha=SCATTER_ALPHA,
            label="Torso valid, invalid nose",
        )
        ax.scatter(*nose[result].T, alpha=SCATTER_ALPHA, label="Valid")

        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.025), fancybox=True, ncol=3
        )

        if not inspection_ax:
            plt.show()

    return result, {
        "nose_within_border": nose_within_border,
        "torso_outside_polygon": torso_outside_polygon,
        "and": result,
    }


def gaze_direction_filter(
    nort_object,
    nose: Sequence[Sequence[float]],
    eye_center: Sequence[Sequence[float]],
    max_radians: float,
    inspect: bool = False,
    inspection_ax: Any = None,
) -> np.ndarray:
    nose, eye_center = np.asarray(nose), np.asarray(eye_center)
    eye_to_nose_unit = unit_vector(nose - eye_center)

    closest_side, idx = closest_line_to_point(
        nort_object.corner_to_corner_vectors, nort_object.perimeter_corners, eye_center
    )

    inner_angles = inner_angle(closest_side, eye_to_nose_unit)

    result = inner_angles <= max_radians

    if inspection_ax is not None or inspect:
        if inspection_ax is None:
            sns.set_theme(style="darkgrid")
            fig, ax = plt.subplots()
        else:
            ax = inspection_ax

        ax.set_title("Gaze direction filter")
        nort_object.plot(ax=ax)

        ax.scatter(*nose[result].T, alpha=SCATTER_ALPHA, label="Valid")
        ax.scatter(*nose[~result].T, alpha=SCATTER_ALPHA, label="Invalid")

        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.025), fancybox=True, ncol=2
        )

        if not ax:
            plt.show()

    return result


def nort_observation(
    nort_object: PolygonalPerimeter,
    eye_center: Sequence[Sequence[float]],
    nose: Sequence[Sequence[float]],
    torso: Sequence[Sequence[float]],
    fps: float,
    perimeter_border_normal_pixel_magnitude: float,
    max_radians_gaze_and_object: float = 0.25 * np.pi,
    inspect: Union[bool, str, PurePath] = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    nort_object: PolygonalPerimeter
    eye_center: Sequence
        Points across time defining the position between the eyes of the animal
    nose: Sequence
        Points across time defining the position of the animal nose
    torso: Sequence
        Points across time defining the central position of the animal torso
    fps: float
        Frames per second in the media used for the respective data source
    perimeter_border_normal_pixel_magnitude
        The magnitude of the normal between the perimeter and the border given in pixels
    max_radians_gaze_and_object: float
        Maximum radians between the gaze vector (eye_centre to nose) and object tangent
    inspect: bool
        If True, will generate and show and inspection figure for the inspection of
        each filter

    Returns
    -------

    """
    eye_center, nose, torso = (
        np.asarray(eye_center),
        np.asarray(nose),
        np.asarray(torso),
    )
    fps = float(fps)
    max_radians_gaze_and_object = float(max_radians_gaze_and_object)

    if inspect:
        if nort_object.inspect_image is not None:
            x, y = nort_object.inspect_image.shape
            fig, axes = plt.subplots(
                nrows=2, ncols=2, figsize=(1.1 * x / 10.0, 1.1 * y / 10.0)
            )
        else:
            fig, axes = plt.subplots(nrows=2, ncols=2)
        fig.gca().invert_yaxis()
        fig.suptitle("Observation cumulative filtration analysis")

        loc_filter_kwargs = {"inspection_ax": axes[0][0]}
        gaze_filter_kwargs = {"inspection_ax": axes[0][1]}
    else:
        loc_filter_kwargs, gaze_filter_kwargs = {}, {}

    location_filtered, loc_analytics = location_filter(
        nort_object,
        nose,
        torso,
        perimeter_border_normal_pixel_magnitude,
        **loc_filter_kwargs,
    )

    gaze_filtered = gaze_direction_filter(
        nort_object,
        nose,
        eye_center,
        max_radians_gaze_and_object,
        **gaze_filter_kwargs,
    )

    semi_true_observations = location_filtered & gaze_filtered

    object_observation = (
        np.zeros_like(semi_true_observations, dtype=bool)
        if np.sum(semi_true_observations) < fps
        else np.array(attention_per_frame(semi_true_observations, fps))
    )

    if inspect:
        for rows in axes:
            for ax in rows:
                nort_object.plot(perimeter_border_normal_pixel_magnitude, ax=ax)

        axes[1][0].set_title("location_filtered & gaze_filtered")
        axes[1][0].scatter(*nose[semi_true_observations].T, alpha=SCATTER_ALPHA)

        axes[1][1].set_title("Object observation")
        axes[1][1].scatter(*nose[object_observation].T, alpha=SCATTER_ALPHA)

        plt.tight_layout()
        if isinstance(inspect, bool):
            plt.show()
        elif isinstance(inspect, str) or isinstance(inspect, PurePath):
            inspect = Path(inspect).resolve()
            if not inspect.parent.exists():
                os.mkdir(inspect.parent)
            plt.savefig(inspect.with_suffix(".jpg"))

    return object_observation, location_filtered, gaze_filtered
