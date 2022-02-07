"""
2D kinematic filters, 3D not supported.
"""
import os
from logging import getLogger
from pathlib import Path, PurePath
from typing import Any, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

from bikipy.feature.angle import inner_angle
from bikipy.math.vector import unit_vector
from bikipy.perimeter.base import PolygonPerimeter
from bikipy.utils.misc import seek_next_file_index

SCATTER_ALPHA = 0.55
logger = getLogger(__name__)


def proximity_filter(
    perimeter: PolygonPerimeter,
    inside_perimeter_border: Sequence[Sequence[float]],
    outside_perimeter: Sequence[Sequence[float]],
    perimeter_border_normal_pixel_magnitude: float,
    inspect: bool = False,
    inspection_ax: Any = None,
) -> Sequence[bool]:
    """
    Filter with respect to proximity rules. (1) The inside_perimeter_border has to be in front of perimeter, but inside the perimeter;
    (2) the outside_perimeter is outside of the perimeter.

    :param perimeter:
    :param inside_perimeter_border: Cartesian coordinates of the inside_perimeter_border
    :param outside_perimeter: Cartesian coordinates of the center of mass
    :param perimeter_border_normal_pixel_magnitude: The magnitude of the normal between the perimeter and the perimeter given in pixels
    :param inspect: If True, generate and view an analytics of the resulting filter
    :param inspection_ax: matplotlib Axes that the inspection plots will (optionally) be saved in
    :type perimeter: PolygonPerimeter
    :type inside_perimeter_border: np.ndarray
    :type outside_perimeter: np.ndarray
    :type perimeter_border_normal_pixel_magnitude: float
    :type inspect: bool
    :type inspection_ax: Any
    :return:
    :rtype: np.ndarray
    """
    # Remove inside_perimeter_border points that aren't inside the perimeter
    inside_perimeter_border = np.asarray(inside_perimeter_border)
    outside_perimeter = np.asarray(outside_perimeter)

    perimeter_border = perimeter.expand(
        perimeter_border_normal_pixel_magnitude=perimeter_border_normal_pixel_magnitude
    )

    inside_perimeter_border_boolean_index = (
        perimeter_border.coordinate_confinement_boolean_index(
            coordinates=inside_perimeter_border
        )
    )
    outside_perimeter_boolean_index = ~perimeter.coordinate_confinement_boolean_index(
        outside_perimeter
    )

    result = inside_perimeter_border_boolean_index & outside_perimeter_boolean_index

    if inspection_ax is not None or inspect:
        if inspection_ax is None:
            sb.set_theme(style="darkgrid")
            fig, ax = plt.subplots()
            if np.any(perimeter.inspect_image):
                ax.imread(perimeter.inspect_image)
        else:
            ax = inspection_ax

        ax.set_title("Location filter")
        perimeter.plot(ax=ax)

        not_result = ~result
        ax.scatter(
            *inside_perimeter_border[
                inside_perimeter_border_boolean_index & not_result
            ].T,
            alpha=SCATTER_ALPHA,
            label="Nose valid, invalid outside_perimeter",
        )
        ax.scatter(
            *inside_perimeter_border[outside_perimeter_boolean_index & not_result].T,
            alpha=SCATTER_ALPHA,
            label="Center of mass valid, invalid inside_perimeter_border",
        )
        ax.scatter(
            *inside_perimeter_border[result].T, alpha=SCATTER_ALPHA, label="Valid"
        )

        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.025), fancybox=True, ncol=3
        )

        if not inspection_ax:
            plt.show()

    return result, (
        inside_perimeter_border_boolean_index,
        outside_perimeter_boolean_index,
    )


def gaze_direction_filter(
    perimeter: PolygonPerimeter,
    gaze_travel_direction_point_label: Sequence[Sequence[float]],
    gaze_start_point_label: Sequence[Sequence[float]],
    max_radians: float,
    inspect: bool = False,
    inspection_ax: Any = None,
):
    gaze_travel_direction_point_label, gaze_start_point_label = np.asarray(
        gaze_travel_direction_point_label
    ), np.asarray(gaze_start_point_label)
    eye_to_nose_vector = gaze_travel_direction_point_label - gaze_start_point_label

    (
        _closest_corner_start_point,
        closest_corner_vectors,
    ) = perimeter.closest_sides_to_coordinates(gaze_start_point_label)

    inner_angles = inner_angle(closest_corner_vectors, eye_to_nose_vector)

    result = inner_angles <= max_radians

    if inspection_ax is not None or inspect:
        if inspection_ax is None:
            sb.set_theme(style="darkgrid")
            fig, ax = plt.subplots()
        else:
            ax = inspection_ax

        ax.set_title("Gaze direction filter")
        perimeter.plot(ax=ax)

        ax.scatter(
            *gaze_travel_direction_point_label[result].T,
            alpha=SCATTER_ALPHA,
            label="Valid",
        )
        ax.scatter(
            *gaze_travel_direction_point_label[~result].T,
            alpha=SCATTER_ALPHA,
            label="Invalid",
        )

        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.025), fancybox=True, ncol=2
        )

        if not ax:
            plt.show()

    return result, closest_corner_vectors


def tolerance_filter(
    boolean_index: Sequence[bool],
    fps: float,
    minimum_seconds_attention: float,
    maximum_seconds_distraction: float = 0.5,
) -> np.ndarray:
    """
    Filters boolean_index with respect to attention. The filter tolerates distraction, and requires
    minimum_seconds_attention to be fulfilled before accepting the sequence as attention.

    :param boolean_index:
    :param fps: Frames per second (fps) of the recording used to generate the data in boolean_index
    :param minimum_seconds_attention: Minimum number of seconds that the sequence has to be True
    for it to be defined as an attention sequence. Filtered sequences will be converted to False.
    :param maximum_seconds_distraction:
    :type boolean_index: np.ndarray
    :type fps: float
    :type minimum_seconds_attention: float
    :return: Boolean index filtered with respect to attention
    :rtype np.ndarray
    """

    boolean_index = np.asarray(boolean_index)

    fps = float(fps)

    distraction_tolerance = round(maximum_seconds_distraction * fps)
    minimum_frames_attention = round(minimum_seconds_attention * fps)

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

            if true_counter == minimum_frames_attention:
                # The first valid index is the index of the first True, i.e. when true_counter was 1
                first_valid_index = i - minimum_frames_attention + 1

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
        and np.sum(attention_boolean_index) >= minimum_frames_attention
    ), (
        f"True: {np.sum(attention_boolean_index)}; fps: {fps}; "
        f"Minimum observation frames: {minimum_frames_attention}"
    )

    return attention_boolean_index


def perimeter_attention(
    perimeter: PolygonPerimeter,
    eye_center: Sequence[Sequence[float]],
    nose: Sequence[Sequence[float]],
    fps: float,
    perimeter_border_normal_pixel_magnitude: float,
    maximum_radians_inter_gaze_perimeter: float = 0.25 * np.pi,
    minimum_seconds_attention: float = 0.5,
    maximum_seconds_distraction: float = 0.5,
    inspect: Union[bool, str, PurePath] = False,
) -> tuple:
    """

    Parameters
    ----------
    perimeter: PolygonPerimeter
    eye_center: Sequence
        Points across time defining the position between the eyes of the animal
    nose: Sequence
        Points across time defining the position of the animal nose
    fps: float
        Frames per second (fps) of the video the data was collected from
    perimeter_border_normal_pixel_magnitude
        The magnitude of the normal between the perimeter and the perimeter given in pixels
    maximum_radians_inter_gaze_perimeter: float
        Maximum radians between the gaze vector (eye_centre to nose) and perimeter tangent
    minimum_seconds_attention
    inspect: bool
        If True, will generate and show and inspection figure for the inspection of
        each filter

    Returns
    -------

    """
    eye_center, nose = np.asarray(eye_center), np.asarray(nose)
    fps = float(fps)
    maximum_radians_inter_gaze_perimeter = float(maximum_radians_inter_gaze_perimeter)

    if inspect:
        if perimeter.inspect_image is None:
            fig, axes = plt.subplots(nrows=2, ncols=2)
        else:
            x, y = perimeter.inspect_image.shape
            fig, axes = plt.subplots(
                nrows=2, ncols=2, figsize=(1.1 * x / 10.0, 1.1 * y / 10.0)
            )

        fig.gca().invert_yaxis()
        fig.suptitle("Observation cumulative filtration analysis")

        loc_filter_kwargs = {"inspection_ax": axes[0][0]}
        gaze_filter_kwargs = {"inspection_ax": axes[0][1]}
    else:
        loc_filter_kwargs, gaze_filter_kwargs = {}, {}

    proximity_filtered, (
        proximity_inside_perimeter_border_boolean_index,
        proximity_outside_perimeter_boolean_index,
    ) = proximity_filter(
        perimeter,
        nose,
        eye_center,
        perimeter_border_normal_pixel_magnitude,
        **loc_filter_kwargs,
    )

    gaze_filtered, gaze_closest_vectors = gaze_direction_filter(
        perimeter,
        nose,
        eye_center,
        maximum_radians_inter_gaze_perimeter,
        **gaze_filter_kwargs,
    )

    semi_true_observations = proximity_filtered & gaze_filtered

    perimeter_observation = (
        np.zeros_like(semi_true_observations, dtype=bool)
        if np.sum(semi_true_observations) < fps
        else np.array(
            tolerance_filter(
                semi_true_observations,
                fps,
                minimum_seconds_attention,
                maximum_seconds_distraction,
            )
        )
    )

    if inspect:
        for rows in axes:
            for ax in rows:
                perimeter.plot_self(
                    plot_kwargs={"ax": ax},
                    perimeter_plot_kwargs={
                        "perimeter_border_normal_pixel_magnitude": perimeter_border_normal_pixel_magnitude
                    },
                )

        axes[1][0].set_title("proximity_filtered & gaze_filtered")
        axes[1][0].scatter(
            *nose[semi_true_observations].T,
            alpha=SCATTER_ALPHA,
        )

        axes[1][1].set_title("BasePerimeter observation")
        axes[1][1].scatter(
            *nose[perimeter_observation].T,
            alpha=SCATTER_ALPHA,
        )

        plt.tight_layout()
        if isinstance(inspect, bool):
            plt.show()
        elif isinstance(inspect, str) or isinstance(inspect, PurePath):
            inspect = Path(inspect).resolve()
            if not inspect.parent.exists():
                os.mkdir(inspect.parent)
            plt.savefig(seek_next_file_index(inspect / f"perimeter_attention.jpg"))

    return (
        perimeter_observation,
        (
            # Arrays for analysing each filter
            proximity_filtered,
            gaze_filtered,
            semi_true_observations,
        ),
        (
            # Arrays for making video
            proximity_inside_perimeter_border_boolean_index,
            proximity_outside_perimeter_boolean_index,
            gaze_closest_vectors,
        ),
    )
