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

from bikipy.feature.angle import inner_angle
from bikipy.math.vector import unit_vector
from bikipy.perimeter.base import Perimeter
from bikipy.utils.misc import seek_next_file_index

SCATTER_ALPHA = 0.55
logger = getLogger(__name__)


def proximity_filter(
    polygonal_perimeter: Perimeter,
    nose: Sequence[Sequence[float]],
    center_eye: Sequence[Sequence[float]],
    perimeter_border_normal_pixel_magnitude: float,
    inspect: bool = False,
    inspection_ax: Any = None,
) -> Sequence[bool]:
    """
    Filter with respect to proximity rules. (1) The nose has to be in front of perimeter, but inside the border;
    (2) the center_eye is outside of the perimeter.

    :param polygonal_perimeter:
    :param nose: Cartesian coordinates of the nose
    :param center_eye: Cartesian coordinates of the center of mass
    :param perimeter_border_normal_pixel_magnitude: The magnitude of the normal between the perimeter and the border given in pixels
    :param inspect: If True, generate and view an analytics of the resulting filter
    :param inspection_ax: matplotlib Axes that the inspection plots will (optionally) be saved in
    :type polygonal_perimeter: Perimeter
    :type nose: np.ndarray
    :type center_eye: np.ndarray
    :type perimeter_border_normal_pixel_magnitude: float
    :type inspect: bool
    :type inspection_ax: Any
    :return:
    :rtype: np.ndarray
    """
    # Remove nose points that aren't inside the perimeter
    nose = np.asarray(nose)
    center_eye = np.asarray(center_eye)

    polygonal_perimeter_border = polygonal_perimeter.border(
        perimeter_border_normal_pixel_magnitude
    )

    nose_within_border = (
        polygonal_perimeter_border.coordinate_confinement_boolean_index(nose)
    )
    center_eye_outside_polygon = (
        ~polygonal_perimeter.coordinate_confinement_boolean_index(center_eye)
    )

    # Find states where the nose is within perimeter while the center_eye is not over perimeter
    result = nose_within_border & center_eye_outside_polygon

    if inspection_ax is not None or inspect:
        if inspection_ax is None:
            sns.set_theme(style="darkgrid")
            fig, ax = plt.subplots()
        else:
            ax = inspection_ax

        ax.set_title("Location filter")
        polygonal_perimeter.plot(ax=ax)

        not_result = ~result
        ax.scatter(
            *nose[nose_within_border & not_result].T,
            alpha=SCATTER_ALPHA,
            label="Nose valid, invalid center_eye",
        )
        ax.scatter(
            *nose[center_eye_outside_polygon & not_result].T,
            alpha=SCATTER_ALPHA,
            label="Center of mass valid, invalid nose",
        )
        ax.scatter(*nose[result].T, alpha=SCATTER_ALPHA, label="Valid")

        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.025), fancybox=True, ncol=3
        )

        if not inspection_ax:
            plt.show()

    return result, {
        "nose_within_border": nose_within_border,
        "center_eye_outside_polygon": center_eye_outside_polygon,
    }


def gaze_direction_filter(
    polygonal_perimeter: Perimeter,
    nose: Sequence[Sequence[float]],
    center_eye: Sequence[Sequence[float]],
    max_radians: float,
    inspect: bool = False,
    inspection_ax: Any = None,
) -> np.ndarray:
    nose, center_eye = np.asarray(nose), np.asarray(center_eye)
    eye_to_nose_unit = unit_vector(nose - center_eye)

    _closest_distance, closest_vector = polygonal_perimeter.closest_sides_to_points(
        center_eye
    )

    inner_angles = inner_angle(closest_vector, eye_to_nose_unit)

    result = inner_angles <= max_radians

    if inspection_ax is not None or inspect:
        if inspection_ax is None:
            sns.set_theme(style="darkgrid")
            fig, ax = plt.subplots()
        else:
            ax = inspection_ax

        ax.set_title("Gaze direction filter")
        polygonal_perimeter.plot(ax=ax)

        ax.scatter(*nose[result].T, alpha=SCATTER_ALPHA, label="Valid")
        ax.scatter(*nose[~result].T, alpha=SCATTER_ALPHA, label="Invalid")

        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.025), fancybox=True, ncol=2
        )

        if not ax:
            plt.show()

    return result


def attention_filter(
    boolean_index: Sequence[bool],
    fps: float,
    minimum_seconds_attention: float = 0.5,
) -> np.ndarray:
    """
    Filters boolean_index with respect to attention. The filter tolerates distraction, and requires
    minimum_seconds_attention to be fulfilled before accepting the sequence as attention.

    :param boolean_index:
    :param fps: Frames per second (fps) of the recording used to generate the data in boolean_index
    :param minimum_seconds_attention: Minimum number of seconds that the sequence has to be True
    for it to be defined as an attention sequence. Filtered sequences will be converted to False.
    :type boolean_index: np.ndarray
    :type fps: float
    :type minimum_seconds_attention: float
    :return: Boolean index filtered with respect to attention
    :rtype np.ndarray
    """

    boolean_index = np.asarray(boolean_index)

    fps = float(fps)
    minimum_seconds_attention = float(minimum_seconds_attention)

    distraction_tolerance = round(fps / 2.0)
    minimum_time_valid_observation = round(minimum_seconds_attention * fps)

    length = boolean_index.shape[0]
    attention_boolean_index = np.zeros(length, dtype=bool)

    first_valid_index = None
    i, valid_frames_within_border, true_counter, consecutive_false = 0, 0, 0, 0
    while True:
        if boolean_index[i]:
            if consecutive_false:
                true_counter += consecutive_false
                consecutive_false = 0
            else:
                true_counter += 1

            if true_counter == minimum_time_valid_observation:
                # The first valid index is the index of the first True, i.e. when true_counter was 1
                first_valid_index = i - minimum_time_valid_observation + 1

        else:
            if first_valid_index is not None:
                if consecutive_false <= distraction_tolerance:
                    consecutive_false += 1
                else:
                    attention_boolean_index[first_valid_index : i + 1] = True
                    valid_frames_within_border += true_counter

                    consecutive_false, true_counter = 0, 0
                    first_valid_index = None

            else:
                true_counter = 0

        i += 1

        if i == length:
            if first_valid_index is not None:
                attention_boolean_index[first_valid_index:] = True
                valid_frames_within_border += true_counter
            break

    if valid_frames_within_border == 0:
        logger.info(f"Subject didn't observe the polygonal perimeter")

        assert not np.any(attention_boolean_index)
        return attention_boolean_index

    assert (
        np.any(attention_boolean_index)
        and np.sum(attention_boolean_index) >= minimum_time_valid_observation
    ), (
        f"True: {np.sum(attention_boolean_index)}; fps: {fps}; "
        f"Minimum observation frames: {minimum_time_valid_observation}"
    )

    return attention_boolean_index


def polygonal_perimeter_attention(
    polygonal_perimeter: Perimeter,
    nose: Sequence[Sequence[float]],
    center_eye: Sequence[Sequence[float]],
    fps: float,
    perimeter_border_normal_pixel_magnitude: float,
    maximum_radians_inter_gaze_perimeter: float = 0.25 * np.pi,
    inspect: Union[bool, str, PurePath] = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    polygonal_perimeter: Perimeter
    center_eye: Sequence
        Points across time defining the position between the eyes of the animal
    nose: Sequence
        Points across time defining the position of the animal nose
    fps: float
        Frames per second (fps) of the video the data was collected from
    perimeter_border_normal_pixel_magnitude
        The magnitude of the normal between the perimeter and the border given in pixels
    maximum_radians_inter_gaze_perimeter: float
        Maximum radians between the gaze vector (eye_centre to nose) and perimeter tangent
    inspect: bool
        If True, will generate and show and inspection figure for the inspection of
        each filter

    Returns
    -------

    """
    center_eye, nose, center_eye = (
        np.asarray(center_eye),
        np.asarray(nose),
        np.asarray(center_eye),
    )
    fps = float(fps)
    maximum_radians_inter_gaze_perimeter = float(maximum_radians_inter_gaze_perimeter)

    if inspect:
        if polygonal_perimeter.inspect_image is None:
            fig, axes = plt.subplots(nrows=2, ncols=2)
        else:
            x, y = polygonal_perimeter.inspect_image.shape
            fig, axes = plt.subplots(
                nrows=2, ncols=2, figsize=(1.1 * x / 10.0, 1.1 * y / 10.0)
            )

        fig.gca().invert_yaxis()
        fig.suptitle("Observation cumulative filtration analysis")

        loc_filter_kwargs = {"inspection_ax": axes[0][0]}
        gaze_filter_kwargs = {"inspection_ax": axes[0][1]}
    else:
        loc_filter_kwargs, gaze_filter_kwargs = {}, {}

    location_filtered, loc_analytics = proximity_filter(
        polygonal_perimeter,
        nose,
        center_eye,
        perimeter_border_normal_pixel_magnitude,
        **loc_filter_kwargs,
    )

    gaze_filtered = gaze_direction_filter(
        polygonal_perimeter,
        nose,
        center_eye,
        maximum_radians_inter_gaze_perimeter,
        **gaze_filter_kwargs,
    )

    semi_true_observations = location_filtered & gaze_filtered

    perimeter_observation = (
        np.zeros_like(semi_true_observations, dtype=bool)
        if np.sum(semi_true_observations) < fps
        else np.array(attention_filter(semi_true_observations, fps))
    )

    if inspect:
        for rows in axes:
            for ax in rows:
                polygonal_perimeter.plot_self(
                    plot_kwargs={"ax": ax},
                    perimeter_plot_kwargs={
                        "perimeter_border_normal_pixel_magnitude": perimeter_border_normal_pixel_magnitude
                    },
                )

        axes[1][0].set_title("location_filtered & gaze_filtered")
        axes[1][0].scatter(*nose[semi_true_observations].T, alpha=SCATTER_ALPHA)

        axes[1][1].set_title("BasePerimeter observation")
        axes[1][1].scatter(*nose[perimeter_observation].T, alpha=SCATTER_ALPHA)

        plt.tight_layout()
        if isinstance(inspect, bool):
            plt.show()
        elif isinstance(inspect, str) or isinstance(inspect, PurePath):
            inspect = Path(inspect).resolve()
            if not inspect.parent.exists():
                os.mkdir(inspect.parent)
            plt.savefig(seek_next_file_index(inspect / f"perimeter_attention.jpg"))

    return perimeter_observation, location_filtered, gaze_filtered
