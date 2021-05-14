"""
Kinematic filters defined in 2D, 3D not supported. The operations
are memory intensive for large datasets.
"""
from logging import getLogger
from typing import Sequence, SupportsFloat, SupportsInt, Union, Any

import matplotlib.pyplot as plt
from seaborn import set_theme
import numpy as np

from bikipy.border.base import PolygonalBorder
from bikipy.feature.angle import counter_clockwise_angel_2d
from bikipy.math.point_in_polygon import points_in_parallelogram
from bikipy.math.vector import closest_line_to_point, unit_vector
from bikipy.utils.misc import read_image

logger = getLogger(__name__)


def location_filter(
    nort_object,
    nose: Sequence[Sequence[SupportsFloat]],
    torso: Sequence[Sequence[SupportsFloat]],
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
    inspect
        If True, generate and view an analytics of the resulting filter
    inspection_image
        Image from the experiment recording used as background in inspection
    inspection_ax
        matplotlib Axes that the inspection plots will (optionally) be saved in

    Returns
    -------

    """
    # Remove nose points that aren't inside the border
    nose = np.asarray(nose)

    nose_within_border = points_in_parallelogram(
        nort_object.border_corners[1],
        nort_object.border_corners[0],
        nort_object.border_corners[2],
        nose,
    )
    torso_outside_polygon = np.logical_not(
        points_in_parallelogram(
            nort_object.perimeter_corners[1],
            nort_object.perimeter_corners[0],
            nort_object.perimeter_corners[2],
            torso,
            inspect_points=False,
        )
    )

    # Find states where the nose is within border while the torso is not over object
    result = np.logical_and(nose_within_border, torso_outside_polygon)

    if inspection_ax is not None or inspect:
        if inspection_ax is None:
            set_theme(style="darkgrid")
            fig, inspection_ax = plt.subplots()
        inspection_ax.set_title("Location filter")

        if inspection_image:
            inspection_ax.imshow(read_image(inspection_image))

        not_result = np.logical_not(result)
        inspection_ax.scatter(*nose[np.logical_and(nose_within_border, not_result)].T)
        inspection_ax.scatter(*nose[np.logical_and(torso_outside_polygon, not_result)].T)
        inspection_ax.scatter(*nose[result].T)

        inspection_ax.legend(("Nose valid, invalid torso", "Torso valid, invalid nose", "Valid"))

        if not inspection_ax:
            plt.show()

    return result


def gaze_direction_filter(
    nort_object,
    nose: Sequence[Sequence[SupportsFloat]],
    eye_center: Sequence[Sequence[SupportsFloat]],
    max_radians: SupportsFloat,
    inspect: bool = False,
    inspection_image: Any = None,
    inspection_ax: Any = None,
):
    nose, eye_center = np.asarray(nose), np.asarray(eye_center)
    eye_to_nose_unit = unit_vector(nose - eye_center)

    closest_side, idx = closest_line_to_point(
        nort_object.side_vectors, nort_object.perimeter_corners, eye_center
    )

    radians = np.abs(counter_clockwise_angel_2d(closest_side, eye_to_nose_unit) - np.pi)

    result = radians <= max_radians

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
    valid_indexes: Sequence[bool], fps: SupportsFloat
) -> np.ndarray:
    valid_indexes = np.asarray(valid_indexes)

    fps = float(fps)

    distraction_tolerance = round(fps / 2)
    minimum_time_valid_observation = round(fps / 3)

    length = valid_indexes.shape[0]
    observation_boolean_indexes = np.full(length, False)

    first_valid_index = None
    i, valid_frames_within_border, true_counter, consecutive_false = 0, 0, 0, 0
    while True:
        if valid_indexes[i]:
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
                    observation_boolean_indexes[first_valid_index : i + 1] = True
                    valid_frames_within_border += true_counter

                    consecutive_false, true_counter = 0, 0
                    first_valid_index = None

            else:
                true_counter = 0

        i += 1

        if i == length:
            if first_valid_index is not None:
                observation_boolean_indexes[first_valid_index:] = True
                valid_frames_within_border += true_counter

            break

    if valid_frames_within_border == 0:
        logger.info(f"Subject didn't observe the nort object")

        assert not np.any(observation_boolean_indexes)
        return observation_boolean_indexes

    assert (
        np.any(observation_boolean_indexes)
        and np.sum(observation_boolean_indexes) >= minimum_time_valid_observation
    ), (
        f"True: {np.sum(observation_boolean_indexes)}; fps: {fps}; "
        f"Minimum observation frames: {minimum_time_valid_observation}"
    )

    return observation_boolean_indexes


def nort_observation(
    nort_object: PolygonalBorder,
    eye_center: Sequence[Sequence[SupportsFloat]],
    nose: Sequence[Sequence[SupportsFloat]],
    torso: Sequence[Sequence[SupportsFloat]],
    fps: Union[SupportsInt, SupportsFloat],
    max_radians_gaze_and_object: SupportsFloat = 1 / 4 * np.pi,
    inspect: bool = False,
    inspection_image: Any = None,
) -> np.ndarray:
    """

    Parameters
    ----------
    nort_object: PolygonalBorder
    eye_center: Sequence
        Points across time defining the position between the eyes of the animal
    nose: Sequence
        Points across time defining the position of the animal nose
    torso: Sequence
        Points across time defining the central position of the animal torso
    fps: float
        Frames per second in the media used for the respective data source
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
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 20))
        loc_filter_kwargs = {"ax": axes[0][0]}
        gaze_filter_kwargs = {"ax": axes[0][1]}
    else:
        loc_filter_kwargs, gaze_filter_kwargs = {}, {}

    quasi_observations = np.logical_and(
        location_filter(nort_object, nose, torso, **loc_filter_kwargs),
        gaze_direction_filter(
            nort_object,
            nose,
            eye_center,
            max_radians_gaze_and_object,
            **gaze_filter_kwargs,
        ),
    )

    object_observation = (
        np.full_like(quasi_observations, False)
        if np.sum(quasi_observations) < fps
        else np.array(attention_span_filter(quasi_observations, fps))
    )

    if inspect:
        if inspection_image is not None:
            inspection_image = read_image(inspection_image)
            for row in axes:
                for ax in row:
                    ax.imshow(inspection_image)

        axes[1][0].set_title("Quasi object observation")
        axes[1][0].scatter(*nose[quasi_observations].T)

        axes[1][1].set_title("Object observation")
        axes[1][1].scatter(*nose[object_observation].T)

        plt.tight_layout()
        plt.show()

    return object_observation
