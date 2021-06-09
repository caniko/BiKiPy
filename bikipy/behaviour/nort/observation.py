"""
2D kinematic filters, 3D not supported.
"""
from logging import getLogger
from typing import Any, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from bikipy.perimeter.base import PolygonalPerimeter
from bikipy.feature.angle import counter_clockwise_angel_2d
from bikipy.math.point_in_polygon import points_in_parallelogram
from bikipy.math.vector import closest_line_to_point, unit_vector
from bikipy.utils.misc import read_image

logger = getLogger(__name__)


def location_filter(
    nort_object,
    nose: Sequence[Sequence[float]],
    torso: Sequence[Sequence[float]],
    perimeter_border_normal_pixel_magnitude: float,
    inspect: bool = False,
    inspection_image: Any = None,
    inspection_ax: Any = None,
) -> Sequence[bool]:
    """

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
    inspection_image
        Image from the experiment recording used as background in inspection
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
            fig, inspection_ax = plt.subplots()
        inspection_ax.set_title("Location filter")

        if inspection_image:
            inspection_ax.imshow(read_image(inspection_image))

        not_result = ~result
        inspection_ax.scatter(*nose[nose_within_border & not_result].T)
        inspection_ax.scatter(*nose[torso_outside_polygon & not_result].T)
        inspection_ax.scatter(*nose[result].T)

        inspection_ax.legend(
            ("Nose valid, invalid torso", "Torso valid, invalid nose", "Valid")
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
    inspection_image: Any = None,
    inspection_ax: Any = None,
) -> np.ndarray:
    nose, eye_center = np.asarray(nose), np.asarray(eye_center)
    eye_to_nose_unit = unit_vector(nose - eye_center)

    closest_side, idx = closest_line_to_point(
        nort_object.corner_to_corner_vectors, nort_object.perimeter_corners, eye_center
    )

    counter_clockwise_rad = counter_clockwise_angel_2d(closest_side, eye_to_nose_unit)
    clockwise_rad = np.abs(counter_clockwise_rad - 2.0 * np.pi)

    result = (counter_clockwise_rad <= max_radians) | (clockwise_rad <= max_radians)

    if inspection_ax is not None or inspect:
        if inspection_ax is None:
            fig, inspection_ax = plt.subplots()
        inspection_ax.set_title("Gaze direction filter")

        inspection_ax.scatter(*nose[result].T)

        if inspection_image:
            inspection_ax.imshow(read_image(inspection_image))

        if not inspection_ax:
            plt.show()

    return result


def attention_span_filter(
    valid_indices: Sequence[bool],
    fps: float,
    minimum_seconds_observing: float = 0.2,
) -> np.ndarray:
    valid_indices = np.asarray(valid_indices)

    fps = float(fps)
    minimum_seconds_observing = float(minimum_seconds_observing)

    distraction_tolerance = round(fps / 2.0)
    minimum_time_valid_observation = round(minimum_seconds_observing * fps)

    length = valid_indices.shape[0]
    observation_boolean_indices = np.full(length, False)

    first_valid_index = None
    i, valid_frames_within_border, true_counter, consecutive_false = 0, 0, 0, 0
    while True:
        if valid_indices[i]:
            true_counter += 1

            if consecutive_false:
                true_counter += consecutive_false - 1  # make up for the previous += 1
                consecutive_false = 0

            if true_counter == minimum_time_valid_observation:
                first_valid_index = i - minimum_time_valid_observation + 1

        else:
            if first_valid_index is not None:
                if consecutive_false <= distraction_tolerance:
                    consecutive_false += 1
                else:
                    observation_boolean_indices[first_valid_index : i + 1] = True
                    valid_frames_within_border += true_counter

                    consecutive_false, true_counter = 0, 0
                    first_valid_index = None

            else:
                true_counter = 0

        i += 1

        if i == length:
            if first_valid_index is not None:
                observation_boolean_indices[first_valid_index:] = True
                valid_frames_within_border += true_counter

            break

    if valid_frames_within_border == 0:
        logger.info(f"Subject didn't observe the nort object")

        assert not np.any(observation_boolean_indices)
        return observation_boolean_indices

    assert (
        np.any(observation_boolean_indices)
        and np.sum(observation_boolean_indices) >= minimum_time_valid_observation
    ), (
        f"True: {np.sum(observation_boolean_indices)}; fps: {fps}; "
        f"Minimum observation frames: {minimum_time_valid_observation}"
    )

    return observation_boolean_indices


def nort_observation(
    nort_object: PolygonalPerimeter,
    eye_center: Sequence[Sequence[float]],
    nose: Sequence[Sequence[float]],
    torso: Sequence[Sequence[float]],
    fps: float,
    perimeter_border_normal_pixel_magnitude: float,
    max_radians_gaze_and_object: float = 1 / 4 * np.pi,
    inspect: bool = False,
    inspection_image: Any = None,
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
    inspection_image: Any

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
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(45, 45))
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
        np.full_like(semi_true_observations, False)
        if np.sum(semi_true_observations) < fps
        else np.array(attention_span_filter(semi_true_observations, fps))
    )

    if inspect:
        if inspection_image is not None:
            inspection_image = read_image(inspection_image)
            for rows in axes:
                for ax in rows:
                    ax.imshow(inspection_image)

        axes[1][0].set_title("Semi true object observation")
        axes[1][0].scatter(*nose[semi_true_observations].T)

        axes[1][1].set_title("Object observation")
        axes[1][1].scatter(*nose[object_observation].T)

        plt.tight_layout()
        plt.show()

    return object_observation, location_filtered, gaze_filtered
