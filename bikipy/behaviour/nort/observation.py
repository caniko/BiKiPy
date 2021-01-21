from logging import getLogger
from typing import Sequence, SupportsFloat, SupportsInt, Union, Any

import matplotlib.pyplot as plt
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
    ax: Any = None,
) -> Sequence[bool]:
    # Remove nose points that aren't inside the border
    nose = np.asanyarray(nose)
    if nort_object.order == 4:
        nose_within_border = points_in_parallelogram(
            nort_object.borders[1],
            nort_object.borders[0],
            nort_object.borders[2],
            nose,
        )
        torso_outside_polygon = np.logical_not(
            points_in_parallelogram(
                nort_object.sides[1],
                nort_object.sides[0],
                nort_object.sides[2],
                torso,
                inspect_points=False,
            )
        )
    else:
        msg = f"Polygon with {nort_object.order} sides is not supported"
        raise NotImplemented(msg)

    # Find states where the nose is within border while the torso is not over object
    result = np.logical_and(nose_within_border, torso_outside_polygon)

    if ax is not None or inspect:
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_title("Location filter")

        if inspection_image:
            ax.imshow(read_image(inspection_image))

        not_result = np.logical_not(result)
        ax.scatter(*nose[np.logical_and(nose_within_border, not_result)].T)
        ax.scatter(*nose[np.logical_and(torso_outside_polygon, not_result)].T)
        ax.scatter(*nose[result].T)

        ax.legend(("Nose valid, invalid torso", "Torso valid, invalid nose", "Valid"))

        if not ax:
            plt.show()

    return result


def gaze_direction_filter(
    nort_object,
    nose: Sequence[Sequence[SupportsFloat]],
    eye_center: Sequence[Sequence[SupportsFloat]],
    max_radians: SupportsFloat,
    inspect: bool = False,
    inspection_image: Any = None,
    ax: Any = None,
):
    nose, eye_center = np.asanyarray(nose), np.asanyarray(eye_center)
    eye_to_nose_unit = unit_vector(nose - eye_center)

    closest_side, idx = closest_line_to_point(
        nort_object.side_vectors, nort_object.sides, eye_center
    )

    radians = np.abs(counter_clockwise_angel_2d(closest_side, eye_to_nose_unit) - np.pi)

    result = radians <= max_radians

    if ax is not None or inspect:
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_title("Gaze direction filter")

        ax.scatter(*nose[result].T)

        if inspection_image:
            ax.imshow(read_image(inspection_image))

        if not ax:
            plt.show()

    return result


def attention_span_filter(
    valid_indexes: Sequence[bool], fps: SupportsFloat
) -> np.ndarray:
    valid_indexes = np.asanyarray(valid_indexes)

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
    # TODO: Remove this before going public!
    inspection_image: Any = "C:/Users/Can/Projects/Neuroscience/bikipy/examples/data/images/nort/B1/habit_1.png",
) -> np.ndarray:
    """

    Parameters
    ----------
    nort_object
    eye_center
        Points across time defining the position between the eyes of the animal
    nose
        Points across time defining the position of the animal nose
    torso
        Points across time defining the central position of the animal torso
    fps
        Frames per second in the media used for the respective data source
    max_radians_gaze_and_object
        Maximum radians between the gaze vector (eye_centre to nose) and object tangent

    Returns
    -------

    """
    eye_center, nose, torso = (
        np.asanyarray(eye_center),
        np.asanyarray(nose),
        np.asanyarray(torso),
    )
    fps = float(fps)
    max_radians_gaze_and_object = float(max_radians_gaze_and_object)
    inspection_image = read_image(inspection_image)

    if inspect:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 20))
    else:
        axes = ((None, None),)

    quasi_observations = np.logical_and(
        location_filter(nort_object, nose, torso, ax=axes[0][0]),
        gaze_direction_filter(
            nort_object, nose, eye_center, max_radians_gaze_and_object, ax=axes[0][1]
        ),
    )

    object_observation = (
        np.full_like(quasi_observations, False)
        if np.sum(quasi_observations) < fps
        else np.array(attention_span_filter(quasi_observations, fps))
    )

    if inspect:
        if inspection_image is not None:
            for row in axes:
                for ax in row:
                    ax.imshow(read_image(inspection_image))

        axes[1][0].set_title("Quasi object observation")
        axes[1][0].scatter(*nose[quasi_observations].T)

        axes[1][1].set_title("Object observation")
        axes[1][1].scatter(*nose[object_observation].T)

        plt.tight_layout()
        plt.show()

    return object_observation
